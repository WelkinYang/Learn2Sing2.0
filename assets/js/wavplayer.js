var wavesurfer = [];

function render(id, selector, url) {

    var domEl = document.createElement('div');
    domEl.setAttribute("id", "t-" + id);
    document.querySelector(selector).appendChild(domEl);

    // timeline
    var timeline = document.createElement('div');
    timeline.setAttribute("id", "time-" + id);
    document.querySelector(selector).appendChild(timeline);

    trackid = "t" + id;

    wavesurfer[trackid] = WaveSurfer.create({
        container: domEl,
        // The color can be either a simple CSS color or a Canvas gradient
        waveColor: '#DCDCDC',
        progressColor: 'hsla(200, 100%, 30%, 0.5)',
        plugins: [
            WaveSurfer.timeline.create({
                container: timeline
            }),
            WaveSurfer.cursor.create({
                showTime: true,
                opacity: 1,
                customShowTimeStyle: {
                    'background-color': '#000',
                    color: '#fff',
                    padding: '2px',
                    'font-size': '10px'
                }
            })
        ]
    });

    // Make sure you reference each method the same way...
    wavesurfer[trackid].drawBuffer();
    wavesurfer[trackid].load(url);

    // controls
    var contorls = document.createElement('div');
    contorls.setAttribute("style", "margin-top:20px");

    var btn_Backward = document.createElement('button');
    btn_Backward.setAttribute("class", "btn")
    btn_Backward.setAttribute("onclick", "wavesurfer['" + trackid + "'].skipBackward()")
    btn_Backward.innerHTML = "<i class='fa fa-step-backward'></i> Backward"
    contorls.appendChild(btn_Backward);

    var btn_play = document.createElement('button');
    btn_play.setAttribute("class", "btn")
    btn_play.setAttribute("onclick", "wavesurfer['" + trackid + "'].playPause()")
    btn_play.innerHTML = "<i class='fa fa-play'></i> Play / <i class='fa fa-pause'></i> Pause"
    contorls.appendChild(btn_play);

    var btn_Forward = document.createElement('button');
    btn_Forward.setAttribute("class", "btn")
    btn_Forward.setAttribute("onclick", "wavesurfer['" + trackid + "'].skipForward()")
    btn_Forward.innerHTML = "<i class='fa fa-step-forward'></i> Forward"
    contorls.appendChild(btn_Forward);

    document.querySelector(selector).appendChild(contorls);

    return wavesurfer[trackid];
}