const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');
const assert = require('node:assert/strict');

// Run the real playback handlers with delayed media metadata. The browser smoke
// tests cover actual playback; this reproduces a race without network timing.
function playbackFixture() {
  const publishedRoot = path.join(__dirname, '../bigvgan-spl');
  const root = fs.existsSync(publishedRoot) ? publishedRoot : path.join(__dirname, '../site');
  const source = fs.readFileSync(path.join(root, 'app.js'), 'utf8').split('async function init()')[0];
  const fixture = JSON.parse(fs.readFileSync(path.join(root, 'samples.json'), 'utf8'));
  class Element {
    constructor() { this.listeners = {}; this.attrs = {}; this.style = {}; this.value = '0'; }
    addEventListener(name, callback) { this.listeners[name] = callback; }
    setAttribute(name, value) { this.attrs[name] = value; }
    focus() {}
  }
  class Audio extends Element {
    constructor() { super(); this.paused = true; this.currentTime = 0; this.readyState = 0; this.ended = false; }
    play() { this.paused = false; return new Promise(() => {}); }
    pause() {
      if (!this.paused) { this.paused = true; this.listeners.pause?.(); }
    }
  }
  const containers = ['v', 'vr', 'vrt', 'reference'].map(id => {
    const container = new Element();
    container.parts = Object.fromEntries(['audio', 'input[type=range]', '.waveform-progress', '.time-display', '.track-error,.reference-error'].map(name => [name, name === 'audio' ? new Audio() : new Element()]));
    if (id === 'reference') container.parts['input[type=range]'] = null;
    container.querySelector = name => container.parts[name];
    container.button = new Element();
    container.button.dataset = { play: id };
    container.button.closest = () => container;
    return container;
  });
  const workspace = new Element();
  const status = new Element();
  const control = new Element();
  workspace.querySelectorAll = name => name === '[data-play]' ? containers.map(c => c.button) : [];
  workspace.querySelector = () => control;
  const context = vm.createContext({ fixture, document: { querySelector: name => name === '#workspace' ? workspace : status } });
  vm.runInContext(source + '\ncollection=fixture;render();', context);
  return { tracks: vm.runInContext('tracks', context), context, status };
}

test('position lock survives two method switches before metadata is ready', () => {
  const { tracks } = playbackFixture();
  tracks[0].audio.readyState = 1;
  tracks[0].audio.currentTime = 3.4;
  tracks[0].button.listeners.click();
  tracks[1].button.listeners.click();
  assert.equal(tracks[1].pendingTime, 3.4);
  tracks[2].button.listeners.click();
  assert.equal(tracks[2].pendingTime, 3.4);
  tracks[2].audio.readyState = 1;
  tracks[2].audio.listeners.loadedmetadata();
  assert.equal(tracks[2].audio.currentTime, 3.4);
});

test('without position lock a new method starts at its own position', () => {
  const { tracks, context } = playbackFixture();
  vm.runInContext('lockPosition=false;', context);
  tracks[0].audio.readyState = 1;
  tracks[0].audio.currentTime = 3.4;
  tracks[0].button.listeners.click();
  tracks[1].button.listeners.click();
  assert.equal(tracks[1].pendingTime, 0);
});

test('a media failure after playback starts announces the error and retry action', () => {
  const { tracks, status } = playbackFixture();
  tracks[0].audio.readyState = 1;
  tracks[0].button.listeners.click();
  tracks[0].audio.listeners.playing();
  assert.match(status.textContent, /Playing V/);
  tracks[0].audio.listeners.error();
  assert.match(status.textContent, /could not load/);
  assert.match(status.textContent, /retry/i);
});
