function handleDrop(e){
  e.preventDefault();
  const dt = new DataTransfer();
  dt.items.add(e.dataTransfer.files[0]);
  document.getElementById('fileInput').files = dt.files;
  document.getElementById('uploadForm').submit();
}

document.getElementById('drop-area')
        .addEventListener('click',()=> document.getElementById('fileInput').click());
