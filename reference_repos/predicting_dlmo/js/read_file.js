function setSummary(minimumTime) {
    var ctx = document.getElementById("summary").innerHTML += ", " + (minimumTime % 24).toFixed(2) + "";
}


function setFilename(filename) {
    var ctx = document.getElementById("summary").innerHTML += "<br>" + filename;
}


function showFile() {

    if (window.File && window.FileReader && window.FileList && window.Blob) {

        var preview = document.getElementById('show-text');

        var files = document.querySelector('input[type=file]').files;
        var ctx = document.getElementById("summary").innerHTML = "<b>Uploaded file name, Predicted DLMO </b><br>";

        for (let j = 0; j < files.length; j++) {
            var file = files[j];
            let filename = file.name;

            var reader = new FileReader();
            var textFile = /text.*/;
            var excelFile = /application.*/;

            console.log(file);
            if (file.type.match(textFile) || file.type.match(excelFile)) {
                reader.onload = function (event) {

                    var worker = new Worker('./js/prep_data.js');
                    worker.onmessage = function (e) {
                        if (typeof e.data === 'number') {
                        } else {
                            const { filename, minimumTime } = e.data;
                            setFilename(filename);
                            setSummary(minimumTime);
                        }
                    };

                    let rawData = event.target.result;

                    worker.postMessage({ rawData, filename });
                }
            } else {
                preview.innerHTML = "<span class='error'>It doesn't seem to be a text file!</span>";
            }
            reader.readAsText(file);
        }
    } else {
        alert("Your browser is too old to support HTML5 File API");
    }
}
