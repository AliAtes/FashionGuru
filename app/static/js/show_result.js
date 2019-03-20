var el = x => document.getElementById(x);

// For image
var dataImage = localStorage.getItem('imgData');
saved_image = el('saved_image');
saved_image.src = "data:image/jpeg;base64," + dataImage;
