const demoRoot = 'assets/demos/';
const demos = {
  strawberry: {
    index: '01 / 04',
    title: 'Select strawberries',
    fastwam: 'fastwam-select-strawberry.mp4',
    dreamwam: 'dreamwam-select-strawberry.mp4',
    fastwamPoster: 'fastwam-select-strawberry.jpg',
    dreamwamPoster: 'dreamwam-select-strawberry.jpg',
  },
  dish: {
    index: '02 / 04',
    title: 'Stack the dish',
    fastwam: 'fastwam-stack-dish.mp4',
    dreamwam: 'dreamwam-stack-dish.mp4',
    fastwamPoster: 'fastwam-stack-dish.jpg',
    dreamwamPoster: 'dreamwam-stack-dish.jpg',
  },
  dual: {
    index: '03 / 04',
    title: 'Dual-object pick',
    fastwam: 'fastwam-dual-box-pick.mp4',
    dreamwam: 'dreamwam-dual-box-pick.mp4',
    fastwamPoster: 'fastwam-dual-box-pick.jpg',
    dreamwamPoster: 'dreamwam-dual-box-pick.jpg',
  },
  blocks: {
    index: '04 / 04',
    title: 'Stack blocks',
    fastwam: 'fastwam-stack-two-blocks.mp4',
    dreamwam: 'dreamwam-stack-two-blocks.mp4',
    fastwamPoster: 'fastwam-stack-two-blocks.jpg',
    dreamwamPoster: 'dreamwam-stack-two-blocks.jpg',
  },
};

const tabs = Array.from(document.querySelectorAll('[data-demo]'));
const fastwamVideo = document.querySelector('#fastwam-video');
const dreamwamVideo = document.querySelector('#dreamwam-video');
const demoIndex = document.querySelector('#demo-index');
const demoTitle = document.querySelector('#demo-title');
const syncPlay = document.querySelector('#sync-play');
const videos = Array.from(document.querySelectorAll('video'));

videos.forEach((video) => {
  video.defaultMuted = true;
  video.muted = true;
  video.volume = 0;
  video.addEventListener('volumechange', () => {
    if (!video.muted) video.muted = true;
    if (video.volume !== 0) video.volume = 0;
  });
});

function selectDemo(name) {
  const demo = demos[name];
  if (!demo || !fastwamVideo || !dreamwamVideo) return;
  fastwamVideo.pause();
  dreamwamVideo.pause();
  fastwamVideo.src = demoRoot + demo.fastwam;
  dreamwamVideo.src = demoRoot + demo.dreamwam;
  fastwamVideo.poster = demoRoot + demo.fastwamPoster;
  dreamwamVideo.poster = demoRoot + demo.dreamwamPoster;
  fastwamVideo.load();
  dreamwamVideo.load();
  demoIndex.textContent = demo.index;
  demoTitle.textContent = demo.title;
  tabs.forEach((tab) => {
    const selected = tab.dataset.demo === name;
    tab.classList.toggle('active', selected);
    tab.setAttribute('aria-selected', String(selected));
  });
}

tabs.forEach((tab) => tab.addEventListener('click', () => selectDemo(tab.dataset.demo)));

if (syncPlay) {
  syncPlay.addEventListener('click', async () => {
    fastwamVideo.currentTime = 0;
    dreamwamVideo.currentTime = 0;
    await Promise.allSettled([fastwamVideo.play(), dreamwamVideo.play()]);
  });
}
