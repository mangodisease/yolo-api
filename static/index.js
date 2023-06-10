video = document.getElementById('video');
source = document.getElementById('source');

function PlayVideo(srcVideo){
  video.pause();
  source.src = srcVideo;
  video.load();
  video.play();
}

function StopVideo(){
  document.getElementById('video').pause();
}

window.onload = () => {
  $("#processing").css("visibility", "hidden");
  $("#sendbutton").click(() => {
    $("#sendbutton").css("visibility", "hidden");
    $("#processing").css("visibility", "visible");
    imagebox = $("#imagebox");
    link = $("#link");
    input = $("#imageinput")[0];
    if (input.files && input.files[0]) {
      let formData = new FormData();
      formData.append("video", input.files[0]);
      $.ajax({
        url: "/detect", // fix this to your liking
        type: "POST",
        data: formData,
        cache: false,
        processData: false,
        contentType: false,
        error: function (data) {
          console.log("upload error", data);
          console.log(data.getAllResponseHeaders());
        },
        success: function (data) {
          console.log(data);
          // bytestring = data["status"];
          // image = bytestring.split("'")[1];
          $("#link").css("visibility", "visible");
          $("#sendbutton").css("visibility", "visible");
          $("#processing").css("visibility", "hidden");
          $("#download").attr("href", "static/video/" + data);
          PlayVideo("static/video/"+ data)
        },
      });
    }
  });
  $("#download").click(() => {
    setTimeout(()=>{
      window.location.reload()
    }, 1500)
  })
};


function readUrl(input) {
  imagebox = $("#imagebox");
  console.log(imagebox);
  console.log("evoked readUrl");
  if (input.files && input.files[0]) {
    let reader = new FileReader();
    reader.onload = function (e) {
      console.log(e.target);

      imagebox.attr("src", e.target.result);
      //   imagebox.height(500);
      //   imagebox.width(800);
    };
    reader.readAsDataURL(input.files[0]);
  }
}


function openCam(e){
  console.log("evoked openCam");
  e.preventDefault();
  console.log("evoked openCam");
  console.log(e);
}