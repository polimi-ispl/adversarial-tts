const workspace = document.querySelector('#workspace');
const status = document.querySelector('#player-status');
const methods = [
  { id: 'v', name: 'V', description: 'Baseline vocoder' },
  { id: 'vr', name: 'VR', description: 'Targets RawNet2' },
  { id: 'vrt', name: 'VRT', description: 'Targets RawNet2 + TSSD' },
];
const icons = {
  play: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5 13 8l-9 5.5Z"/></svg>',
  pause: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 3h3v10H4zm5 0h3v10H9z"/></svg>',
  download: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 3v9m-3-3 3 3 3-3M4 13v4h12v-4"/></svg>',
};
let collection;
let speakerIndex = 0;
let promptIndex = 0;
let lockPosition = true;
let tracks = [];
let activeTrack = null;
let generation = 0;

function time(seconds) {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  return `${Math.floor(safe / 60)}:${Math.floor(safe % 60).toString().padStart(2, '0')}`;
}

function wave(peaks, height = 36) {
  return peaks.map((peak, i) => {
    const h = Math.max(1.4, Math.min(height - 2, peak * height * 1.7));
    return `<rect x="${i * 4}" y="${(height - h) / 2}" width="2" height="${h}" rx="1" fill="currentColor"/>`;
  }).join('');
}

function pauseAll() {
  for (const track of tracks) track.audio.pause();
}

function render(focusSelector) {
  pauseAll();
  generation += 1;
  const currentGeneration = generation;
  tracks = [];
  activeTrack = null;
  const speaker = collection.speakers[speakerIndex];
  const prompt = collection.prompts[promptIndex];
  const sample = speaker.samples[promptIndex];
  workspace.innerHTML = `
    <aside class="speaker-panel" aria-label="Choose a speaker">
      <p class="panel-label">Choose a voice</p>
      <div class="speaker-list" role="group" aria-label="Speakers">
        ${collection.speakers.map((s, i) => `${i === 4 ? '<p class="group-divider">Unseen voices</p>' : ''}<button class="speaker-button" data-speaker="${i}" aria-pressed="${i === speakerIndex}" aria-label="${s.name}, ${s.voice.toLowerCase()}, ${s.group.toLowerCase()}"><span class="voice-avatar" aria-hidden="true">${i < 4 ? s.id.replace(/\D/g, '') : s.id === 'gorou' ? 'G' : 'Y'}</span><span><span class="speaker-name">${s.name}</span><span class="speaker-detail">${s.voice} · ${i < 4 ? 'Seen' : 'Unseen'}</span></span></button>`).join('')}
      </div>
      <p class="speaker-footnote">6 voices, 3 prompts.<br>The same text across all vocoders.</p>
    </aside>
    <div class="comparison-panel">
      <div class="comparison-top"><p class="selected-speaker"><strong>${speaker.name}</strong><span class="pill">${speaker.group}</span></p><label class="prompt-control" for="prompt-select">Prompt<select id="prompt-select">${collection.prompts.map((p, i) => `<option value="${i}" ${i === promptIndex ? 'selected' : ''}>${p.label}</option>`).join('')}</select></label></div>
      <div class="prompt-copy"><p>“${prompt.excerpt}”</p><small>Prompt excerpt from the original filename</small></div>
      <div class="playback-toolbar"><p class="playback-hint">Play a track to start comparing.</p><label class="lock-control"><input id="position-lock" type="checkbox" ${lockPosition ? 'checked' : ''}> Keep playback position</label></div>
      <div class="track-list" role="group" aria-label="Vocoder comparison for ${speaker.name}, ${prompt.label}">
        ${methods.map(m => `<div class="track track-${m.id}" data-track="${m.id}"><button class="play-button" data-play="${m.id}" aria-label="Play ${m.name}, ${speaker.name}, ${prompt.label}" aria-pressed="false">${icons.play}</button><div class="track-name"><strong>${m.name}</strong><span>${m.description}</span></div><div class="waveform-control"><svg class="waveform" viewBox="0 0 384 36" preserveAspectRatio="none" aria-hidden="true">${wave(sample[m.id].peaks)}</svg><span class="waveform-progress"></span><input type="range" min="0" max="${sample[m.id].duration}" step="0.01" value="0" aria-label="Seek ${m.name} audio" aria-valuetext="0:00 of ${time(sample[m.id].duration)}"><span class="time-display">0:00 / ${time(sample[m.id].duration)}</span></div><a class="download-link" href="${sample[m.id].src}" download="${speaker.id}-${prompt.id}-${m.id}.wav" aria-label="Download ${m.name}, ${speaker.name}, ${prompt.label}">${icons.download}</a><p class="track-error" hidden></p><audio preload="metadata" src="${sample[m.id].src}"></audio></div>`).join('')}
      </div>
      <div class="reference-row"><button class="reference-button" data-play="reference" aria-label="Play speaker reference for ${speaker.name}" aria-pressed="false">${icons.play}<span>Speaker reference</span><span>${time(speaker.reference.duration)}</span></button><p class="reference-note">Original voice sample. Different text from the generated prompt.</p><p class="reference-error" hidden></p><audio preload="metadata" src="${speaker.reference.src}"></audio></div>
    </div>`;
  workspace.setAttribute('aria-busy', 'false');

  for (const button of workspace.querySelectorAll('[data-speaker]')) {
    button.addEventListener('click', () => {
      if (speakerIndex === Number(button.dataset.speaker)) return;
      speakerIndex = Number(button.dataset.speaker);
      render(`[data-speaker="${speakerIndex}"]`);
      status.textContent = `${collection.speakers[speakerIndex].name} selected. Playback reset.`;
    });
  }
  workspace.querySelector('#prompt-select').addEventListener('change', event => {
    promptIndex = Number(event.target.value);
    render('#prompt-select');
    status.textContent = `${collection.prompts[promptIndex].label} selected. Playback reset.`;
  });
  workspace.querySelector('#position-lock').addEventListener('change', event => { lockPosition = event.target.checked; });

  for (const button of workspace.querySelectorAll('[data-play]')) {
    const id = button.dataset.play;
    const container = id === 'reference' ? button.closest('.reference-row') : button.closest('.track');
    const audio = container.querySelector('audio');
    const slider = container.querySelector('input[type=range]');
    const progress = container.querySelector('.waveform-progress');
    const display = container.querySelector('.time-display');
    const error = container.querySelector('.track-error,.reference-error');
    const name = id === 'reference' ? 'speaker reference' : methods.find(m => m.id === id).name;
    const duration = id === 'reference' ? speaker.reference.duration : sample[id].duration;
    const track = { id, audio, button, pendingTime: null };
    tracks.push(track);
    function updateButton(playing) {
      if (generation !== currentGeneration) return;
      button.setAttribute('aria-pressed', String(playing));
      button.setAttribute('aria-label', `${playing ? 'Pause' : 'Play'} ${name}, ${speaker.name}${id === 'reference' ? '' : ', ' + prompt.label}`);
      if (id === 'reference') button.innerHTML = `${playing ? icons.pause : icons.play}<span>Speaker reference</span><span>${time(duration)}</span>`;
      else button.innerHTML = playing ? icons.pause : icons.play;
    }
    function updateTime() {
      if (!slider || generation !== currentGeneration) return;
      slider.value = audio.currentTime;
      slider.setAttribute('aria-valuetext', `${time(audio.currentTime)} of ${time(duration)}`);
      display.textContent = `${time(audio.currentTime)} / ${time(duration)}`;
      progress.style.width = `${Math.min(100, audio.currentTime / duration * 100)}%`;
    }
    function seek(seconds) {
      const value = Math.max(0, Math.min(seconds, duration - 0.01));
      if (audio.readyState >= 1) audio.currentTime = value;
      else track.pendingTime = value;
    }
    audio.addEventListener('loadedmetadata', () => {
      if (track.pendingTime !== null) { audio.currentTime = track.pendingTime; track.pendingTime = null; }
    });
    audio.addEventListener('playing', () => {
      if (generation !== currentGeneration || activeTrack !== track) { audio.pause(); return; }
      updateButton(true);
      status.textContent = `Playing ${name}, ${speaker.name}.`;
      workspace.querySelector('.playback-hint').textContent = id === 'reference' ? 'Listening to the speaker reference.' : `Now playing ${name}. Switch a vocoder to compare.`;
    });
    audio.addEventListener('pause', () => updateButton(false));
    audio.addEventListener('ended', () => {
      updateButton(false);
      if (generation === currentGeneration && activeTrack === track) {
        workspace.querySelector('.playback-hint').textContent = 'Track finished. Play again or choose another vocoder.';
      }
    });
    audio.addEventListener('timeupdate', updateTime);
    audio.addEventListener('seeked', updateTime);
    audio.addEventListener('error', () => {
      if (generation !== currentGeneration) return;
      error.hidden = false;
      error.textContent = 'Audio could not load. Select Play to retry, or open the original collection below.';
      status.textContent = `${name} audio could not load. Select Play to retry.`;
      updateButton(false);
    });
    if (slider) slider.addEventListener('input', () => {
      seek(Number(slider.value));
      display.textContent = `${time(Number(slider.value))} / ${time(duration)}`;
      progress.style.width = `${Number(slider.value) / duration * 100}%`;
      slider.setAttribute('aria-valuetext', `${time(Number(slider.value))} of ${time(duration)}`);
    });
    button.addEventListener('click', () => {
      if (activeTrack === track && !audio.paused) {
        audio.pause();
        status.textContent = `Paused ${name}.`;
        return;
      }
      let startTime = audio.ended ? 0 : (track.pendingTime ?? audio.currentTime);
      if (lockPosition && activeTrack && activeTrack !== track && id !== 'reference' && activeTrack.id !== 'reference') {
        startTime = activeTrack.audio.ended ? 0 : (activeTrack.pendingTime ?? activeTrack.audio.currentTime);
      }
      pauseAll();
      activeTrack = track;
      error.hidden = true;
      if (audio.error) audio.load();
      seek(startTime);
      status.textContent = `Loading ${name}.`;
      updateButton(true);
      audio.play().catch(playError => {
        if (generation !== currentGeneration || activeTrack !== track || playError.name === 'AbortError') return;
        updateButton(false);
        error.hidden = false;
        error.textContent = 'Playback could not start. Select Play to retry, or download the recording.';
        status.textContent = 'Playback could not start.';
      });
    });
  }
  if (focusSelector) workspace.querySelector(focusSelector)?.focus({ preventScroll: true });
}

async function init() {
  try {
    const response = await fetch(new URL('samples.json', import.meta.url));
    if (!response.ok) throw new Error(`Collection response ${response.status}`);
    collection = await response.json();
    render();
    for (const svg of document.querySelectorAll('.hero-wave')) {
      svg.innerHTML = wave(collection.speakers[0].samples[0][svg.dataset.method].peaks, 110);
      svg.setAttribute('viewBox', '0 0 384 110');
    }
  } catch (error) {
    workspace.setAttribute('aria-busy', 'false');
    workspace.innerHTML = '<p class="initial-state" role="alert">The audio collection could not load. Please reload this page or visit the <a href="https://h0yjmbl9to9.feishu.cn/docx/FH7mdd1dco6xDkxFlhPchve8nyh">original audio collection</a>.</p>';
    console.error('Could not initialize audio collection:', error);
  }
}

window.addEventListener('pagehide', pauseAll);
init();
