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
  avatarModel: 'miku', // 'miku' | 'female' | 'male'
  gender: 'female',
  hair: 'Hair_Miku_Twintails',
  outfit: 'Outfit_Miku_Full',
  shoes: 'Shoes_Miku_Boots',
  accessories: new Set([
    'Outfit_Miku_Sleeves',
    'Accessory_Miku_Headset',
    'Accessory_Miku_Tie',
    'Accessory_CatEars'
  ]),
  eyeStyle: 'moe',
  expression: 'smile',
  colors: {
    skin: '#fff0ea',
    hair: '#ffffff', // default white (preserves original texture color)
    iris: '#2288ff',
    outfit_primary: '#ffffff' // default white (preserves original texture color)
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
let currentLoadedModelType = null;
let modelBounds = { center: new THREE.Vector3(0, 0, 0), size: new THREE.Vector3(1, 2, 1), maxDim: 2 };
let dirLight, ambLight, fillLight, rimLight, bottomLight, shadowPlane, grid;

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

  // 4. OrbitControls (Full 360-degree spherical rotation)
  orbitControls = new OrbitControls(camera, renderer.domElement);
  orbitControls.enableDamping = true;
  orbitControls.dampingFactor = 0.08;
  orbitControls.target.set(0, 0.05, 0);
  orbitControls.minPolarAngle = 0.01;
  orbitControls.maxPolarAngle = Math.PI - 0.01;
  orbitControls.minDistance = 0.4;
  orbitControls.maxDistance = 5.0;

  // 5. Lighting (Anime Cel Studio Lighting)
  ambLight = new THREE.AmbientLight(0xfff0f5, 0.85);
  scene.add(ambLight);

  const keyLight = new THREE.DirectionalLight(0xffffff, 1.25);
  keyLight.position.set(3, 5, 4);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.set(2048, 2048);
  keyLight.shadow.bias = -0.0005;
  scene.add(keyLight);

  fillLight = new THREE.DirectionalLight(0x818cf8, 0.55);
  fillLight.position.set(-3, 3, -2);
  scene.add(fillLight);

  rimLight = new THREE.DirectionalLight(0xec4899, 0.45);
  rimLight.position.set(0, 4, -4);
  scene.add(rimLight);

  bottomLight = new THREE.DirectionalLight(0x38bdf8, 0.35);
  bottomLight.position.set(0, -4, 2);
  scene.add(bottomLight);

  // Ground Grid & Shadow Plane
  const shadowGeo = new THREE.PlaneGeometry(100, 100);
  const shadowMat = new THREE.ShadowMaterial({ opacity: 0.3 });
  shadowPlane = new THREE.Mesh(shadowGeo, shadowMat);
  shadowPlane.rotation.x = -Math.PI / 2;
  shadowPlane.position.y = -1.0;
  shadowPlane.receiveShadow = true;
  scene.add(shadowPlane);

  grid = new THREE.GridHelper(40, 40, 0x4f46e5, 0x1e293b);
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
  eyeTexture.flipY = false;
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
// -----------------------------------------------------------------------------
// Modular Model Loader
// -----------------------------------------------------------------------------
function loadModularModel(targetModelType = avatarConfig.avatarModel) {
  const loader = new GLTFLoader();
  const isMiku = (targetModelType === 'miku');
  const url = isMiku ? 'miku_modular.glb?v=2.0' : 'anime_avatar_modular.glb?v=2.0';

  exportModal.style.display = 'flex';
  modalTitle.textContent = isMiku ? '載入初音未來模組化紙娃娃中...' : '載入原創紙娃娃模型中...';
  modalDesc.textContent = isMiku 
    ? '初始化 SEGA 高精手繪部件、動態物理骨架與動作庫...' 
    : '初始化 16 款動漫幾何模組與 3D 骨架中...';

  loader.load(
    url,
    (gltf) => {
      // 1. Clean up old model
      if (model) {
        scene.remove(model);
        model = null;
      }
      if (mixer) {
        mixer.stopAllAction();
        mixer = null;
      }
      clipsMap.clear();
      allMeshes.clear();

      // 2. Add new model
      model = gltf.scene;
      scene.add(model);
      currentLoadedModelType = targetModelType;

      model.traverse((child) => {
        if (child.isMesh) {
          child.castShadow = true;
          child.receiveShadow = true;
          allMeshes.set(child.name, child);

          // Transparent eye decal material for procedural avatar
          if (child.name === 'Face_Eyes') {
            child.material = new THREE.MeshBasicMaterial({
              map: eyeTexture,
              transparent: true,
              alphaTest: 0.02,
              depthWrite: false,
              polygonOffset: true,
              polygonOffsetFactor: -4,
              side: THREE.DoubleSide
            });
            child.renderOrder = 10;
          }
        }
      });

      console.log(`Loaded ${targetModelType} model with ${allMeshes.size} modular parts.`);

      // 3. Compute dynamic bounding box
      const box = new THREE.Box3().setFromObject(model);
      modelBounds.center = box.getCenter(new THREE.Vector3());
      modelBounds.size = box.getSize(new THREE.Vector3());
      modelBounds.maxDim = Math.max(modelBounds.size.x, modelBounds.size.y, modelBounds.size.z);

      // Adjust shadow plane, grid, and bottom light based on model scale
      if (shadowPlane) shadowPlane.position.y = box.min.y;
      if (grid) grid.position.y = box.min.y + 0.001;
      if (bottomLight) bottomLight.position.set(0, box.min.y - 2, 2);

      // 4. Setup Animation Mixer
      if (gltf.animations && gltf.animations.length > 0) {
        mixer = new THREE.AnimationMixer(model);
        gltf.animations.forEach((clip) => {
          const action = mixer.clipAction(clip);
          clipsMap.set(clip.name.toLowerCase(), action);
        });
        console.log('Embedded avatar animations:', Array.from(clipsMap.keys()));
      }

      // 5. Apply component configuration
      applyAvatarConfiguration();

      // 6. Camera Auto-Frame
      setCameraPreset('full');

      // 7. Play default action
      playAnimation(currentAnimName);

      // 8. Hide modal
      exportModal.style.display = 'none';
      onWindowResize();
    },
    undefined,
    (err) => {
      console.error('Error loading model:', err);
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

  if (currentLoadedModelType === 'miku') {
    // === MIKU MODULAR PARTS ===
    // 1. Hair
    const hasHair = (cfg.hair === 'Hair_Miku_Twintails');
    setMeshVisibility('Hair_Miku_Twintails', hasHair);

    // 2. Head & Facial Features
    setMeshVisibility('Head_Miku', true);
    setMeshVisibility('Face_Eyes_Miku', true);
    setMeshVisibility('Face_Mouth_Miku', true);

    // 3. Outfits
    if (cfg.outfit === 'Outfit_Miku_Full') {
      setMeshVisibility('Outfit_Miku_Top', true);
      setMeshVisibility('Outfit_Miku_Skirt', true);
    } else if (cfg.outfit === 'Outfit_Miku_Top') {
      setMeshVisibility('Outfit_Miku_Top', true);
      setMeshVisibility('Outfit_Miku_Skirt', false);
    } else if (cfg.outfit === 'Outfit_Miku_Skirt') {
      setMeshVisibility('Outfit_Miku_Top', false);
      setMeshVisibility('Outfit_Miku_Skirt', true);
    } else {
      setMeshVisibility('Outfit_Miku_Top', false);
      setMeshVisibility('Outfit_Miku_Skirt', false);
    }

    // 4. Sleeves
    setMeshVisibility('Outfit_Miku_Sleeves', cfg.accessories.has('Outfit_Miku_Sleeves'));

    // 5. Boots
    setMeshVisibility('Shoes_Miku_Boots', cfg.shoes === 'Shoes_Miku_Boots');

    // 6. Accessories
    setMeshVisibility('Accessory_Miku_Headset', cfg.accessories.has('Accessory_Miku_Headset'));
    setMeshVisibility('Accessory_Miku_Tie', cfg.accessories.has('Accessory_Miku_Tie'));
  } else {
    // === PROCEDURAL AVATAR PARTS ===
    const isFemale = cfg.gender === 'female';
    setMeshVisibility('Body_Female', isFemale);
    setMeshVisibility('Body_Male', !isFemale);

    const hairStyles = ['Hair_Twintails', 'Hair_Bob', 'Hair_Hime', 'Hair_Spiky', 'Hair_Parted'];
    hairStyles.forEach(h => setMeshVisibility(h, h === cfg.hair));

    const outfits = ['Outfit_Sailor', 'Outfit_Blazer', 'Outfit_Hoodie', 'Outfit_Casual'];
    outfits.forEach(o => setMeshVisibility(o, o === cfg.outfit));

    const shoes = ['Shoes_Loafers', 'Shoes_Sneakers'];
    shoes.forEach(s => setMeshVisibility(s, s === cfg.shoes));

    setMeshVisibility('Accessory_Glasses', cfg.accessories.has('Accessory_Glasses'));
    setMeshVisibility('Accessory_CatEars', cfg.accessories.has('Accessory_CatEars'));
    setMeshVisibility('Face_Eyes', true);

    updateFaceTexture();
  }

  // Apply Colors & Proportions
  applyColors();
  applyProportions();
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
    if (!mesh.material || mesh.name === 'Face_Eyes') return;
    const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];

    mats.forEach((mat) => {
      const matName = mat.name || '';
      const mName = mesh.name;

      if (mName.includes('Hair')) {
        mat.color = new THREE.Color(cfg.colors.hair);
        mat.needsUpdate = true;
      } else if (mName.includes('Outfit') || mName.includes('Dress') || matName.includes('Outfit')) {
        mat.color = new THREE.Color(cfg.colors.outfit_primary);
        mat.needsUpdate = true;
      } else if (mName.includes('Body') || matName.includes('Skin')) {
        mat.color = new THREE.Color(cfg.colors.skin);
        mat.needsUpdate = true;
      }
    });
  });
}

function applyProportions() {
  if (!model) return;
  const p = avatarConfig.proportions;

  // Height and shoulder scaling
  model.scale.set(p.shoulder, p.height, 1.0);

  // Head bone scaling
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
// Camera Presets (Auto-scaled with model)
// -----------------------------------------------------------------------------
function setCameraPreset(preset) {
  if (!orbitControls || !model) return;
  const c = modelBounds.center;
  const maxD = modelBounds.maxDim;

  if (preset === 'full') {
    orbitControls.target.copy(c);
    camera.position.set(c.x, c.y + maxD * 0.05, c.z + maxD * 1.35);
    orbitControls.minDistance = maxD * 0.2;
    orbitControls.maxDistance = maxD * 5.0;
  } else if (preset === 'face') {
    const headTarget = new THREE.Vector3(c.x, c.y + maxD * 0.38, c.z);
    orbitControls.target.copy(headTarget);
    camera.position.set(c.x, c.y + headTarget.y * 0.05 + maxD * 0.38, c.z + maxD * 0.38);
    orbitControls.minDistance = maxD * 0.1;
  } else if (preset === 'torso') {
    const torsoTarget = new THREE.Vector3(c.x, c.y + maxD * 0.16, c.z);
    orbitControls.target.copy(torsoTarget);
    camera.position.set(c.x, c.y + maxD * 0.18, c.z + maxD * 0.72);
    orbitControls.minDistance = maxD * 0.15;
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
          const nameTag = (currentLoadedModelType === 'miku') ? 'miku_custom' : `avatar_${avatarConfig.gender}`;
          link.download = `${nameTag}_${timeStr}.glb`;
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
function syncUiToConfig() {
  const cfg = avatarConfig;

  // Single select cards
  document.querySelectorAll('.opt-card:not(.toggle-card)').forEach((card) => {
    const type = card.dataset.type;
    const val = card.dataset.val;

    if (type === 'avatar_model') card.classList.toggle('active', val === cfg.avatarModel);
    else if (type === 'hair') card.classList.toggle('active', val === cfg.hair);
    else if (type === 'outfit') card.classList.toggle('active', val === cfg.outfit);
    else if (type === 'shoes') card.classList.toggle('active', val === cfg.shoes);
    else if (type === 'eye_style') card.classList.toggle('active', val === cfg.eyeStyle);
  });

  // Toggle cards
  document.querySelectorAll('.toggle-card').forEach((card) => {
    const toggle = card.dataset.toggle;
    card.classList.toggle('active', cfg.accessories.has(toggle));
  });

  // Color dots
  document.querySelectorAll('.color-palette').forEach((palette) => {
    const target = palette.dataset.target;
    const targetColor = cfg.colors[target];
    palette.querySelectorAll('.color-dot').forEach((dot) => {
      dot.classList.toggle('active', dot.dataset.color.toLowerCase() === (targetColor || '').toLowerCase());
    });
  });
}

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

  // Option Cards (Single Selection: avatar_model, hair, outfit, shoes, eye_style)
  document.querySelectorAll('.opt-card:not(.toggle-card)').forEach(card => {
    card.addEventListener('click', () => {
      const type = card.dataset.type;
      const val = card.dataset.val;

      // Deselect siblings
      document.querySelectorAll(`.opt-card[data-type="${type}"]`).forEach(c => c.classList.remove('active'));
      card.classList.add('active');

      if (type === 'avatar_model') {
        avatarConfig.avatarModel = val;
        if (val === 'miku') {
          avatarConfig.gender = 'female';
          avatarConfig.hair = 'Hair_Miku_Twintails';
          avatarConfig.outfit = 'Outfit_Miku_Full';
          avatarConfig.shoes = 'Shoes_Miku_Boots';
          avatarConfig.accessories.add('Outfit_Miku_Sleeves');
          avatarConfig.accessories.add('Accessory_Miku_Headset');
          avatarConfig.accessories.add('Accessory_Miku_Tie');
          avatarConfig.colors.hair = '#ffffff';
          avatarConfig.colors.outfit_primary = '#ffffff';
        } else {
          avatarConfig.gender = val;
          avatarConfig.hair = 'Hair_Twintails';
          avatarConfig.outfit = 'Outfit_Sailor';
          avatarConfig.shoes = 'Shoes_Loafers';
          avatarConfig.colors.hair = '#33c7df';
          avatarConfig.colors.outfit_primary = '#1e293b';
        }
        syncUiToConfig();
        loadModularModel(val === 'miku' ? 'miku' : 'procedural');
        return;
      } else if (type === 'hair') {
        avatarConfig.hair = val;
      } else if (type === 'outfit') {
        avatarConfig.outfit = val;
      } else if (type === 'shoes') {
        avatarConfig.shoes = val;
      } else if (type === 'eye_style') {
        avatarConfig.eyeStyle = val;
      }

      applyAvatarConfiguration();
    });
  });

  // Toggle Cards (Accessories: CatEars, Glasses, Sleeves, Headset, Tie)
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
