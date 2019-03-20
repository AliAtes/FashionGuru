var el = x => document.getElementById(x);

// For Image
var dataImage = localStorage.getItem('imgData');
saved_image = el('saved_image');
saved_image.src = "data:image/jpeg;base64," + dataImage;
alert('saved_image.src: ' + saved_image.src);
