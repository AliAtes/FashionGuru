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
classes = ['Blouse', 'Blazer', 'Button-Down', 'Bomber', 'Anorak', 'Tee', 'Tank', 'Top', 'Sweater', 'Flannel', 'Hoodie', 'Cardigan', 'Jacket', 'Henley', 'Poncho', 'Jersey', 'Turtleneck', 'Parka', 'Peacoat', 'Halter', 'Skirt', 'Shorts', 'Jeans', 'Joggers', 'Sweatpants', 'Jeggings', 'Cutoffs', 'Sweatshorts', 'Leggings', 'Culottes', 'Chinos', 'Trunks', 'Sarong', 'Gauchos', 'Jodhpurs', 'Capris', 'Dress', 'Romper', 'Coat', 'Kimono', 'Jumpsuit', 'Robe', 'Caftan', 'Kaftan', 'Coverup', 'Onesie']

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
    logging.warning("data[img]: " + str(data["img"]))
    bytes = base64.b64decode(img_bytes)
    return predict_from_bytes(bytes)

def predict_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    
    _,_,losses = learn.predict(img)
    predictions = sorted(zip(classes, map(float, losses)), key=lambda p: p[1], reverse=True)
    logging.warning("predictions[0]: " + str(predictions[0][0]))
    
    http = urllib3.PoolManager()
    page = http.request('GET', 'https://www.google.com/search?q=' + str(predictions[0][0]) + '&tbm=shop')
    
    soup = BeautifulSoup(page.data, 'html.parser')
    
    all_cards_html = ""	
    for tag in soup.find_all("div", attrs={'class': 'pslires'}):
    	link = tag.div.a['href']
    	image = tag.div.img['src']
    	info = tag.div.img['alt']
    	shop = tag.findChildren()[6].getText()
    	amount = tag.findChildren()[3].div.getText()
    	all_cards_html += ("<!-- predictions: { " + "[" + predictions[0][0] + " - " predictions[0][1] + "], " + "[" + predictions[1][0] + " - " predictions[1][1] + "], " + "[" + predictions[2][0] + " - " predictions[2][1] + "]"    + " }--><div class=\"col-6 col-sm-6 col-md-4 col-lg-3\" style=\"margin-top:20px;\"><a href=\"" + link + "\"><table style=\"text-align:center;\"><tr><td><img src=\"" + image + "\" /></td></tr><tr><td>" + info + " | " + shop + "</td></tr><tr><td>" + amount + "</td></tr></table></a></div>")
    
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
