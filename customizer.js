import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';

// Global Studio State
let scene, camera, renderer, orbitControls;
let model = null, mixer = null, skeletonHelper = null;
let animClock = new THREE.Clock();
let animSpeed = 1.0;
let isPlaying = true;
let currentAnimName = 'idle';

// Meshes Dictionary
const allMeshes = new Map(); // name -> Mesh
const clipsMap = new Map();  // name lowercase -> Action

// Customizer Configuration
const avatarConfig = {
  gender: 'female',
  hair: 'Hair_Twintails',
  outfit: 'Outfit_Sailor',
  shoes: 'Shoes_Loafers',
  accessories: new Set(['Accessory_CatEars']),
  eyeStyle: 'moe',
  expression: 'smile',
  colors: {
    skin: '#fff0ea',
    hair: '#33c7df',
    iris: '#2288ff',
    outfit_primary: '#1e293b'
  },
  proportions: {
    height: 1.0,
    head: 1.0,
    shoulder: 1.0
  },
  showOutline: true
};

// Canvas Face / Eye Texture
let eyeCanvas, eyeCtx, eyeTexture;

// DOM Elements
const canvas = document.getElementById('canvas3d');
const exportModal = document.getElementById('export-modal');
const modalTitle = document.getElementById('modal-title');
const modalDesc = document.getElementById('modal-desc');

// Initialize Studio
init();

function init() {
  // 1. Three.js Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0f1d);

  // 2. Camera Setup
  const w = canvas.clientWidth || (window.innerWidth - 380);
  const h = canvas.clientHeight || (window.innerHeight - 60);
  const aspect = w / h;
  camera = new THREE.PerspectiveCamera(40, aspect, 0.1, 100);
  camera.position.set(0, 0.1, 2.7);

  // 3. Renderer Setup
  renderer = new THREE.WebGLRenderer({
    canvas: canvas,
    antialias: true,
    preserveDrawingBuffer: true
  });
  renderer.setSize(w, h);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;

  // 4. OrbitControls
  orbitControls = new OrbitControls(camera, renderer.domElement);
  orbitControls.enableDamping = true;
  orbitControls.dampingFactor = 0.08;
  orbitControls.target.set(0, 0.05, 0);
  orbitControls.maxPolarAngle = Math.PI / 2 + 0.05;
  orbitControls.minDistance = 0.4;
  orbitControls.maxDistance = 5.0;

  // 5. Lighting (Anime Cel Studio Lighting)
  const ambLight = new THREE.AmbientLight(0xfff0f5, 0.85);
  scene.add(ambLight);

  const keyLight = new THREE.DirectionalLight(0xffffff, 1.25);
  keyLight.position.set(3, 5, 4);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.set(2048, 2048);
  keyLight.shadow.bias = -0.0005;
  scene.add(keyLight);

  const fillLight = new THREE.DirectionalLight(0x818cf8, 0.55);
  fillLight.position.set(-3, 3, -2);
  scene.add(fillLight);

  const rimLight = new THREE.DirectionalLight(0xec4899, 0.45);
  rimLight.position.set(0, 4, -4);
  scene.add(rimLight);

  // Ground Grid & Shadow Plane
  const shadowGeo = new THREE.PlaneGeometry(10, 10);
  const shadowMat = new THREE.ShadowMaterial({ opacity: 0.3 });
  const shadowPlane = new THREE.Mesh(shadowGeo, shadowMat);
  shadowPlane.rotation.x = -Math.PI / 2;
  shadowPlane.position.y = -1.0;
  shadowPlane.receiveShadow = true;
  scene.add(shadowPlane);

  const grid = new THREE.GridHelper(8, 24, 0x4f46e5, 0x1e293b);
  grid.position.y = -0.999;
  scene.add(grid);

  // 6. Setup Canvas Face Texture
  initFaceCanvas();

  // 7. Load Modular Anime Master GLB
  loadModularModel();

  // 8. Event Listeners
  setupEventListeners();
  window.addEventListener('resize', onWindowResize);

  // 9. Start Loop
  animate();
}

// -----------------------------------------------------------------------------
// Face / Eye Canvas Procedural Texture Generator
// -----------------------------------------------------------------------------
function initFaceCanvas() {
  eyeCanvas = document.createElement('canvas');
  eyeCanvas.width = 1024;
  eyeCanvas.height = 512;
  eyeCtx = eyeCanvas.getContext('2d');

  eyeTexture = new THREE.CanvasTexture(eyeCanvas);
  eyeTexture.colorSpace = THREE.SRGBColorSpace;
  eyeTexture.generateMipmaps = true;

  updateFaceTexture();
}

function updateFaceTexture() {
  if (!eyeCtx) return;
  const ctx = eyeCtx;
  const w = eyeCanvas.width;
  const h = eyeCanvas.height;

  ctx.clearRect(0, 0, w, h);

  // Draw Left Eye (cx=320) & Right Eye (cx=704)
  drawEye(ctx, 320, 260, 1.0);
  drawEye(ctx, 704, 260, -1.0);

  // Draw Mouth (cx=512, cy=440)
  drawMouth(ctx, 512, 440);

  // Draw Cheeks Blush
  drawBlush(ctx, 240, 340);
  drawBlush(ctx, 784, 340);

  eyeTexture.needsUpdate = true;
}

function drawEye(ctx, cx, cy, flipX) {
  ctx.save();
  ctx.translate(cx, cy);
  ctx.scale(flipX, 1);

  const style = avatarConfig.eyeStyle;
  const irisColor = avatarConfig.colors.iris;

  // 1. Sclera (Eye White)
  ctx.beginPath();
  ctx.ellipse(0, 0, 85, 110, 0, 0, Math.PI * 2);
  ctx.fillStyle = '#ffffff';
  ctx.fill();

  // Sclera top shadow
  ctx.beginPath();
  ctx.ellipse(0, -50, 85, 50, 0, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(180, 195, 220, 0.45)';
  ctx.fill();

  // 2. Iris (Gradient)
  let irisW = 62, irisH = 92;
  if (style === 'cool') { irisW = 52; irisH = 80; }
  else if (style === 'gentle') { irisW = 65; irisH = 88; }
  else if (style === 'cat') { irisW = 56; irisH = 94; }

  const grad = ctx.createLinearGradient(0, -irisH, 0, irisH);
  grad.addColorStop(0, '#0a1020');
  grad.addColorStop(0.35, irisColor);
  grad.addColorStop(0.85, '#ffffff');

  ctx.beginPath();
  ctx.ellipse(0, 5, irisW, irisH, 0, 0, Math.PI * 2);
  ctx.fillStyle = grad;
  ctx.fill();

  // 3. Pupil
  ctx.beginPath();
  if (style === 'cat') {
    ctx.ellipse(0, 5, 14, 60, 0, 0, Math.PI * 2); // Slit pupil
  } else {
    ctx.ellipse(0, 5, 26, 38, 0, 0, Math.PI * 2);
  }
  ctx.fillStyle = '#060814';
  ctx.fill();

  // 4. Star & Specular Highlights
  ctx.fillStyle = '#ffffff';
  // Big top-left highlight
  ctx.beginPath();
  ctx.ellipse(-22, -28, 22, 28, -0.3, 0, Math.PI * 2);
  ctx.fill();

  // Small bottom-right highlight
  ctx.beginPath();
  ctx.arc(24, 32, 11, 0, Math.PI * 2);
  ctx.fill();

  if (style === 'moe') {
    // Additional cute sparkle stars
    ctx.beginPath();
    ctx.arc(-10, 42, 6, 0, Math.PI * 2);
    ctx.arc(15, -45, 5, 0, Math.PI * 2);
    ctx.fill();
  }

  // 5. Eyeliner & Lashes
  ctx.strokeStyle = '#181820';
  ctx.lineWidth = 14;
  ctx.lineCap = 'round';

  ctx.beginPath();
  if (style === 'cool') {
    ctx.moveTo(-75, -60);
    ctx.quadraticCurveTo(0, -115, 88, -90); // Sharp wing
  } else if (style === 'gentle') {
    ctx.moveTo(-80, -95);
    ctx.quadraticCurveTo(0, -105, 80, -65); // Drooping wing
  } else {
    ctx.moveTo(-80, -75);
    ctx.quadraticCurveTo(0, -118, 80, -75);
  }
  ctx.stroke();

  // Outer lash wing
  ctx.lineWidth = 8;
  ctx.beginPath();
  ctx.moveTo(70, -82);
  ctx.lineTo(95, -96);
  ctx.stroke();

  // Lower eyelash accent
  ctx.lineWidth = 6;
  ctx.beginPath();
  ctx.moveTo(-45, 100);
  ctx.quadraticCurveTo(15, 115, 55, 95);
  ctx.stroke();

  // 6. Eyebrow (Tapered)
  ctx.strokeStyle = avatarConfig.colors.hair;
  ctx.lineWidth = 12;
  ctx.beginPath();
  ctx.moveTo(-70, -150);
  ctx.quadraticCurveTo(0, -175, 75, -145);
  ctx.stroke();

  ctx.restore();
}

function drawMouth(ctx, cx, cy) {
  ctx.save();
  ctx.translate(cx, cy);

  ctx.strokeStyle = '#8a3a4b';
  ctx.fillStyle = '#e85d75';
  ctx.lineWidth = 7;
  ctx.lineCap = 'round';

  const expr = avatarConfig.expression;
  ctx.beginPath();
  if (expr === 'wink') {
    ctx.arc(0, -5, 22, 0.2, Math.PI - 0.2);
  } else if (expr === 'pout') {
    ctx.arc(0, 0, 14, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fill();
    ctx.restore();
    return;
  } else {
    // Normal cute smile
    ctx.arc(0, -8, 26, 0.15, Math.PI - 0.15);
  }
  ctx.stroke();
  ctx.restore();
}

function drawBlush(ctx, cx, cy) {
  ctx.save();
  ctx.translate(cx, cy);

  const grad = ctx.createRadialGradient(0, 0, 5, 0, 0, 60);
  grad.addColorStop(0, 'rgba(255, 120, 150, 0.45)');
  grad.addColorStop(1, 'rgba(255, 120, 150, 0.0)');

  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(0, 0, 60, 0, Math.PI * 2);
  ctx.fill();

  // Two cute diagonal anime blush lines
  ctx.strokeStyle = 'rgba(230, 80, 120, 0.6)';
  ctx.lineWidth = 4;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(-15, 5); ctx.lineTo(-5, -15);
  ctx.moveTo(5, 5);   ctx.lineTo(15, -15);
  ctx.stroke();

  ctx.restore();
}

// -----------------------------------------------------------------------------
// Modular Model Loader
// -----------------------------------------------------------------------------
function loadModularModel() {
  const loader = new GLTFLoader();
  const url = 'anime_avatar_modular.glb?v=1.1';

  exportModal.style.display = 'flex';
  modalTitle.textContent = '載入日系卡漫紙娃娃中...';
  modalDesc.textContent = '初始化 16 款動漫模組與 3D 骨架中，請稍候';

  loader.load(
    url,
    (gltf) => {
      model = gltf.scene;
      scene.add(model);

      allMeshes.clear();
      model.traverse((child) => {
        if (child.isMesh) {
          child.castShadow = true;
          child.receiveShadow = true;
          allMeshes.set(child.name, child);

          // Apply special transparent eye decal material
          if (child.name === 'Face_Eyes') {
            child.material = new THREE.MeshBasicMaterial({
              map: eyeTexture,
              transparent: true,
              alphaTest: 0.05,
              depthWrite: false,
              polygonOffset: true,
              polygonOffsetFactor: -4
            });
          }
        }
      });

      console.log(`Loaded Anime Avatar Studio Model with ${allMeshes.size} modular parts.`);

      // Setup Animation Mixer
      if (gltf.animations && gltf.animations.length > 0) {
        mixer = new THREE.AnimationMixer(model);
        gltf.animations.forEach((clip) => {
          const action = mixer.clipAction(clip);
          clipsMap.set(clip.name.toLowerCase(), action);
        });
        console.log('Embedded avatar animations:', Array.from(clipsMap.keys()));
      }

      // Apply initial component selection and colors
      applyAvatarConfiguration();

      // Play Idle Animation
      playAnimation(currentAnimName);

      // Hide loading modal
      exportModal.style.display = 'none';
      onWindowResize();
    },
    undefined,
    (err) => {
      console.error('Error loading anime avatar modular model:', err);
      modalTitle.textContent = '模型載入失敗';
      modalDesc.textContent = err.message || '請確認網路連線或重新整理頁面。';
    }
  );
}

// -----------------------------------------------------------------------------
// Component Visibility & Color Application
// -----------------------------------------------------------------------------
function applyAvatarConfiguration() {
  if (!model) return;

  const cfg = avatarConfig;

  // 1. Gender / Base Body
  const isFemale = cfg.gender === 'female';
  setMeshVisibility('Body_Female', isFemale);
  setMeshVisibility('Body_Male', !isFemale);

  // 2. Hairstyles
  const hairStyles = ['Hair_Twintails', 'Hair_Bob', 'Hair_Hime', 'Hair_Spiky', 'Hair_Parted'];
  hairStyles.forEach(h => setMeshVisibility(h, h === cfg.hair));

  // 3. Outfits
  const outfits = ['Outfit_Sailor', 'Outfit_Blazer', 'Outfit_Hoodie', 'Outfit_Casual'];
  outfits.forEach(o => setMeshVisibility(o, o === cfg.outfit));

  // 4. Shoes
  const shoes = ['Shoes_Loafers', 'Shoes_Sneakers'];
  shoes.forEach(s => setMeshVisibility(s, s === cfg.shoes));

  // 5. Accessories
  setMeshVisibility('Accessory_Glasses', cfg.accessories.has('Accessory_Glasses'));
  setMeshVisibility('Accessory_CatEars', cfg.accessories.has('Accessory_CatEars'));

  // 6. Eyes always visible
  setMeshVisibility('Face_Eyes', true);

  // 7. Apply Colors to Materials
  applyColors();

  // 8. Apply Body Proportions
  applyProportions();

  // 9. Update Eye Texture
  updateFaceTexture();
}

function setMeshVisibility(name, visible) {
  const mesh = allMeshes.get(name);
  if (mesh) {
    mesh.visible = visible;
  }
}

function applyColors() {
  const cfg = avatarConfig;

  allMeshes.forEach((mesh) => {
    if (!mesh.material) return;
    const matName = mesh.material.name || '';

    if (matName.includes('Skin') || mesh.name.includes('Body')) {
      mesh.material.color = new THREE.Color(cfg.colors.skin);
    } else if (matName.includes('Hair') || mesh.name.includes('Hair')) {
      mesh.material.color = new THREE.Color(cfg.colors.hair);
    } else if (matName.includes('Outfit') || mesh.name.includes('Outfit')) {
      mesh.material.color = new THREE.Color(cfg.colors.outfit_primary);
    }
  });
}

function applyProportions() {
  if (!model) return;
  const p = avatarConfig.proportions;

  // Height scaling
  model.scale.set(p.shoulder, p.height, 1.0);

  // Head bone scale
  const headBone = model.getObjectByName('Head');
  if (headBone) {
    headBone.scale.set(p.head, p.head, p.head);
  }
}

// -----------------------------------------------------------------------------
// Animation Control
// -----------------------------------------------------------------------------
function playAnimation(animName) {
  currentAnimName = animName;
  if (!mixer) return;

  const key = animName.toLowerCase();
  clipsMap.forEach((act) => act.stop());

  if (clipsMap.has(key)) {
    const action = clipsMap.get(key);
    action.reset().fadeIn(0.2).play();
  }
}

function togglePlayPause(play) {
  if (play === undefined) isPlaying = !isPlaying;
  else isPlaying = play;

  const btn = document.getElementById('btn-play-pause');
  if (isPlaying) {
    btn.textContent = '⏸️';
    if (mixer) mixer.timeScale = animSpeed;
  } else {
    btn.textContent = '▶️';
    if (mixer) mixer.timeScale = 0;
  }
}

// -----------------------------------------------------------------------------
// Camera Presets
// -----------------------------------------------------------------------------
function setCameraPreset(preset) {
  if (!orbitControls) return;

  if (preset === 'full') {
    camera.position.set(0, 0.1, 2.7);
    orbitControls.target.set(0, 0.05, 0);
  } else if (preset === 'face') {
    camera.position.set(0, 0.82, 0.70);
    orbitControls.target.set(0, 0.80, 0);
  } else if (preset === 'torso') {
    camera.position.set(0, 0.45, 1.45);
    orbitControls.target.set(0, 0.35, 0);
  }
  orbitControls.update();
}

// -----------------------------------------------------------------------------
// Export GLB Engine
// -----------------------------------------------------------------------------
function exportCustomGLB() {
  if (!model) return;

  exportModal.style.display = 'flex';
  modalTitle.textContent = '正在生成並匯出自訂 GLB 模型...';
  modalDesc.textContent = '打包當前服裝、髮型、五官與骨架中，請稍候';

  setTimeout(() => {
    try {
      const exporter = new GLTFExporter();

      // Filter: only export visible meshes and the skeleton
      const exportScene = new THREE.Scene();
      const clonedModel = model.clone(true);

      // Remove invisible meshes from clone
      const toRemove = [];
      clonedModel.traverse((child) => {
        if (child.isMesh && !child.visible) {
          toRemove.push(child);
        }
      });
      toRemove.forEach((c) => c.parent.remove(c));

      exportScene.add(clonedModel);

      exporter.parse(
        exportScene,
        (glbArrayBuffer) => {
          const blob = new Blob([glbArrayBuffer], { type: 'model/gltf-binary' });
          const url = URL.createObjectURL(blob);
          const link = document.createElement('a');
          const timeStr = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
          link.download = `anime_avatar_${avatarConfig.gender}_${timeStr}.glb`;
          link.href = url;
          link.click();
          URL.revokeObjectURL(url);

          exportModal.style.display = 'none';
        },
        (err) => {
          console.error('GLTF Export Error:', err);
          modalTitle.textContent = '匯出失敗';
          modalDesc.textContent = '請檢查瀏覽器控制台日誌。';
          setTimeout(() => exportModal.style.display = 'none', 2000);
        },
        {
          binary: true,
          onlyVisible: true,
          animations: []
        }
      );
    } catch (e) {
      console.error('Export exception:', e);
      exportModal.style.display = 'none';
    }
  }, 100);
}

// -----------------------------------------------------------------------------
// UI Event Handlers
// -----------------------------------------------------------------------------
function setupEventListeners() {
  // Category Tabs
  document.querySelectorAll('.cat-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.cat-tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.panel-section').forEach(s => s.classList.remove('active'));

      tab.classList.add('active');
      const targetSec = document.getElementById(`sec-${tab.dataset.cat}`);
      if (targetSec) targetSec.classList.add('active');
    });
  });

  // Option Cards (Single Selection: gender, hair, outfit, shoes, eye_style)
  document.querySelectorAll('.opt-card:not(.toggle-card)').forEach(card => {
    card.addEventListener('click', () => {
      const type = card.dataset.type;
      const val = card.dataset.val;

      // Deselect siblings
      document.querySelectorAll(`.opt-card[data-type="${type}"]`).forEach(c => c.classList.remove('active'));
      card.classList.add('active');

      if (type === 'gender') avatarConfig.gender = val;
      else if (type === 'hair') avatarConfig.hair = val;
      else if (type === 'outfit') avatarConfig.outfit = val;
      else if (type === 'shoes') avatarConfig.shoes = val;
      else if (type === 'eye_style') avatarConfig.eyeStyle = val;

      applyAvatarConfiguration();
    });
  });

  // Toggle Cards (Accessories: CatEars, Glasses)
  document.querySelectorAll('.toggle-card').forEach(card => {
    card.addEventListener('click', () => {
      const toggle = card.dataset.toggle;
      const isActive = avatarConfig.accessories.has(toggle);

      if (isActive) {
        avatarConfig.accessories.delete(toggle);
        card.classList.remove('active');
      } else {
        avatarConfig.accessories.add(toggle);
        card.classList.add('active');
      }
      applyAvatarConfiguration();
    });
  });

  // Expression Buttons
  document.querySelectorAll('[data-expr]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-expr]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      avatarConfig.expression = btn.dataset.expr;
      updateFaceTexture();
    });
  });

  // Color Palettes
  document.querySelectorAll('.color-palette').forEach(palette => {
    const target = palette.dataset.target;
    palette.querySelectorAll('.color-dot').forEach(dot => {
      dot.addEventListener('click', () => {
        palette.querySelectorAll('.color-dot').forEach(d => d.classList.remove('active'));
        dot.classList.add('active');

        const color = dot.dataset.color;
        if (target === 'skin') avatarConfig.colors.skin = color;
        else if (target === 'hair') avatarConfig.colors.hair = color;
        else if (target === 'iris') avatarConfig.colors.iris = color;
        else if (target === 'outfit_primary') avatarConfig.colors.outfit_primary = color;

        applyColors();
        updateFaceTexture();
      });
    });
  });

  // Body Proportion Sliders
  const sliderHeight = document.getElementById('slider-body-height');
  const sliderHead = document.getElementById('slider-body-head');
  const sliderShoulder = document.getElementById('slider-body-shoulder');
  const valHeight = document.getElementById('val-body-height');
  const valHead = document.getElementById('val-body-head');
  const valShoulder = document.getElementById('val-body-shoulder');

  sliderHeight.addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
    avatarConfig.proportions.height = v;
    valHeight.textContent = `${v.toFixed(2)}x`;
    applyProportions();
  });

  sliderHead.addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
    avatarConfig.proportions.head = v;
    valHead.textContent = `${v.toFixed(2)}x`;
    applyProportions();
  });

  sliderShoulder.addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
    avatarConfig.proportions.shoulder = v;
    valShoulder.textContent = `${v.toFixed(2)}x`;
    applyProportions();
  });

  document.getElementById('btn-reset-proportions').addEventListener('click', () => {
    avatarConfig.proportions.height = 1.0;
    avatarConfig.proportions.head = 1.0;
    avatarConfig.proportions.shoulder = 1.0;
    sliderHeight.value = 1.0; valHeight.textContent = '1.00x';
    sliderHead.value = 1.0; valHead.textContent = '1.00x';
    sliderShoulder.value = 1.0; valShoulder.textContent = '1.00x';
    applyProportions();
  });

  // Camera Presets Toolbar
  document.querySelectorAll('.cam-btn[data-cam]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.cam-btn[data-cam]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      setCameraPreset(btn.dataset.cam);
    });
  });

  const outlineBtn = document.getElementById('btn-toggle-outline');
  if (outlineBtn) {
    outlineBtn.addEventListener('click', () => {
      avatarConfig.showOutline = !avatarConfig.showOutline;
      outlineBtn.classList.toggle('active', avatarConfig.showOutline);
      allMeshes.forEach(mesh => {
        if (mesh.material && !mesh.name.includes('Eyes')) {
          mesh.material.roughness = avatarConfig.showOutline ? 0.8 : 0.4;
        }
      });
    });
  }

  document.getElementById('btn-reset-cam').addEventListener('click', () => {
    setCameraPreset('full');
  });

  // Animation Buttons Toolbar
  document.querySelectorAll('.anim-btn[data-anim]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.anim-btn[data-anim]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      playAnimation(btn.dataset.anim);
    });
  });

  document.getElementById('btn-play-pause').addEventListener('click', () => togglePlayPause());

  // Random Avatar Button
  document.getElementById('btn-random-avatar').addEventListener('click', randomizeAvatar);

  // Export GLB Button
  document.getElementById('btn-export-glb').addEventListener('click', exportCustomGLB);
}

function randomizeAvatar() {
  const hairs = ['Hair_Twintails', 'Hair_Bob', 'Hair_Hime', 'Hair_Spiky', 'Hair_Parted'];
  const outfits = ['Outfit_Sailor', 'Outfit_Blazer', 'Outfit_Hoodie', 'Outfit_Casual'];
  const eyes = ['moe', 'cool', 'gentle', 'cat'];
  const hairColors = ['#33c7df', '#ff85a2', '#ffd255', '#26262a', '#6d4834', '#e4e6eb', '#a56de2'];
  const irisColors = ['#2288ff', '#22cc88', '#ff3366', '#ffaa00', '#9944ff', '#333333'];
  const outfitColors = ['#1e293b', '#f8fafc', '#be123c', '#0f766e', '#3b82f6', '#18181b'];

  avatarConfig.hair = hairs[Math.floor(Math.random() * hairs.length)];
  avatarConfig.outfit = outfits[Math.floor(Math.random() * outfits.length)];
  avatarConfig.eyeStyle = eyes[Math.floor(Math.random() * eyes.length)];
  avatarConfig.colors.hair = hairColors[Math.floor(Math.random() * hairColors.length)];
  avatarConfig.colors.iris = irisColors[Math.floor(Math.random() * irisColors.length)];
  avatarConfig.colors.outfit_primary = outfitColors[Math.floor(Math.random() * outfitColors.length)];

  applyAvatarConfiguration();
}

function onWindowResize() {
  const w = canvas.clientWidth || (window.innerWidth - 380);
  const h = canvas.clientHeight || (window.innerHeight - 60);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}

function animate() {
  requestAnimationFrame(animate);

  const delta = Math.min(animClock.getDelta(), 0.1);
  if (isPlaying && mixer) {
    mixer.update(delta * animSpeed);
  }

  orbitControls.update();
  renderer.render(scene, camera);
}
