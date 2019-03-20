from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO
from fastai.vision import *
import base64
import json
import urllib3
from bs4 import BeautifulSoup
import logging

model_file_url = 'https://github.com/AliAtes/DeepFashionKTE/blob/master/app/models/model.pth?raw=true'
model_file_name = 'model'
classes = ['bluz', 'blazer ceket', 'düğmeli giysi', 'bomber ceket', 'anorak', 'tshirt', 'sporcu atleti', 'üst', 'süveter', 'flanel gömlek', 'svetşört', 'uzun hırka', 'mont', 'triko tişört', 'panço', 'jarse', 'balıkçı yaka', 'parka', 'palto', 'askılı', 'etek', 'şort', 'kot', 'pantolon eşofman', 'eşofman altı', 'dar kesim pantolon', 'kot şort', 'Sweatshorts', 'tayt', 'pantolon etek', 'chino pantolon', 'erkek mayosu', 'sarong', 'bol paça kısa pantolon', 'binicilik pantolonu', 'kapri', 'elbise', 'tulum', 'kaban', 'kimono', 'bağcıklı tulum', 'sabahlık', 'kaftan', 'kaftan', 'şal elbise', 'pijama kostüm']

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(model_file_url, path/'models'/f'{model_file_name}.pth')
    data_bunch = ImageDataBunch.single_from_classes(path, classes, ds_tfms=get_transforms(), size=150).normalize(imagenet_stats)
    learn = create_cnn(data_bunch, models.resnet34, pretrained=False)
    learn.load(model_file_name)
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

PREDICTION_FILE_SRC = path/'static'/'predictions.txt'

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    img_bytes = (data["img"])
    radios = str(data["options"])
    logging.warning("data[img]: " + str(data["img"]) + " radios: " + radios)
    bytes = base64.b64decode(img_bytes)
    return predict_from_bytes(bytes, radios)

def predict_from_bytes(bytes, radios):
    img = open_image(BytesIO(bytes))
    
    _,_,losses = learn.predict(img)
    predictions = sorted(zip(classes, map(float, losses)), key=lambda p: p[1], reverse=True)
    logging.warning("predictions[0]: " + str(predictions[0][0]))
    
    http = urllib3.PoolManager()
    
    if(radios == "erkek" && str(predictions[0][0]) == "elbise")
        radios = "takım elbise"
    else if(radios == "erkek" && str(predictions[0][0]) == "bağcıklı tulum")
        radios = "jumpsuit"
    
    logging.warning("Radios+predictions: " + radios + " " + str(predictions[0][0]))
    
    page = http.request('GET', 'https://www.google.com/search?q=' + radios + " " + str(predictions[0][0]) + '&tbm=shop')
    
    soup = BeautifulSoup(page.data, 'html.parser')
    
    all_cards_html = ""	
    for tag in soup.find_all("div", attrs={'class': 'pslires'}):
    	link = tag.div.a['href']
    	image = tag.div.img['src']
    	info = tag.div.img['alt']
    	shop = tag.findChildren()[6].getText()
    	amount = tag.findChildren()[3].div.getText()
    	all_cards_html += ("<div class=\"col-6 col-sm-6 col-md-4 col-lg-3 col-centered text-center card\"><div><a href=\"" + link + "\"><img style=\"width:100%; margin-bottom:5px;\" src=\"" + image + "\" /></a></div><div>" + info + "<br />" + shop + "<br />" + amount + "</div></div>")
    
    all_cards_html += ("<!-- predictions: { " + "[" + predictions[0][0] + " - " + predictions[0][1] + "], " + "[" + predictions[1][0] + " - " + predictions[1][1] + "], " + "[" + predictions[2][0] + " - " + predictions[2][1] + "]" + " }-->")
    logging.warning("all_cards_html: " + all_cards_html)
    
    result_html1 = path/'static'/'result1.html'
    result_html2 = path/'static'/'result2.html'
    result_html = str(result_html1.open().read() + all_cards_html + result_html2.open().read())
    return HTMLResponse(result_html)

@app.route("/")
def form(request):
    index_html = path/'static'/'index.html'
    return HTMLResponse(index_html.open().read())

if __name__ == "__main__":
    if "serve" in sys.argv: uvicorn.run(app, host="0.0.0.0", port=8080)
