import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import * as SkeletonUtils from 'three/addons/utils/SkeletonUtils.js';

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
let modelBounds = { center: new THREE.Vector3(0, 10, 0), size: new THREE.Vector3(1, 20, 1), maxDim: 20 };
let dirLight, ambLight, fillLight, rimLight, bottomLight, shadowPlane, grid;
const bonesMap = new Map();
const initialBoneRotations = new Map();
let animTime = 0;

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
  camera = new THREE.PerspectiveCamera(40, aspect, 0.1, 500);
  camera.position.set(0, 11.5, 28);

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
  orbitControls.target.set(0, 10, 0);
  orbitControls.minPolarAngle = 0.01;
  orbitControls.maxPolarAngle = Math.PI - 0.01;
  orbitControls.minDistance = 0.1;
  orbitControls.maxDistance = 200.0;

  // 5. Lighting (Anime Cel Studio Lighting)
  ambLight = new THREE.AmbientLight(0xfff0f5, 0.85);
  scene.add(ambLight);

  dirLight = new THREE.DirectionalLight(0xffffff, 1.25);
  dirLight.position.set(3, 15, 12);
  dirLight.castShadow = true;
  dirLight.shadow.mapSize.set(2048, 2048);
  dirLight.shadow.bias = -0.0005;
  scene.add(dirLight);

  fillLight = new THREE.DirectionalLight(0x818cf8, 0.55);
  fillLight.position.set(-10, 10, -5);
  scene.add(fillLight);

  rimLight = new THREE.DirectionalLight(0xec4899, 0.45);
  rimLight.position.set(0, 15, -12);
  scene.add(rimLight);

  bottomLight = new THREE.DirectionalLight(0x38bdf8, 0.35);
  bottomLight.position.set(0, -5, 5);
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
  syncUiToConfig();
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
function loadModularModel(targetModelType = avatarConfig.avatarModel) {
  const loader = new GLTFLoader();
  const isMiku = (targetModelType === 'miku');
  const isMint = (targetModelType === 'mint');
  let url = 'anime_avatar_modular.glb?v=2.0';
  let title = '載入原創紙娃娃模型中...';
  let desc = '初始化 16 款動漫幾何模組與 3D 骨架中...';

  if (isMiku) {
    url = 'miku_modular.glb?v=3.0';
    title = '載入初音未來模組化紙娃娃中...';
    desc = '初始化 SEGA 高精手繪部件、多款日系假髮與動作庫...';
  } else if (isMint) {
    url = 'mint_modular.glb?v=2.0';
    title = '載入薄荷 Mint (Neverness To Everness) 中...';
    desc = '初始化 2D Flat 動漫渲、Unlit Emission 材質與 24 秒官方靈動展示舞步...';
  }

  exportModal.style.display = 'flex';
  modalTitle.textContent = title;
  modalDesc.textContent = desc;

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
      bonesMap.clear();
      initialBoneRotations.clear();

      // 2. Add new model
      model = gltf.scene;
      scene.add(model);
      currentLoadedModelType = targetModelType;

      model.traverse((child) => {
        if (child.isMesh) {
          allMeshes.set(child.name, child);

          if (isMint) {
            // Apply 2D Anime Flat Shading instructions:
            // "To avoid realistic 3D shading or weird shadows, set Shadow Mode to None for all materials"
            child.castShadow = false;
            child.receiveShadow = false;
            if (child.material) {
              const mats = Array.isArray(child.material) ? child.material : [child.material];
              mats.forEach(mat => {
                mat.transparent = true;
                mat.alphaTest = 0.05;
                mat.depthWrite = true;
              });
            }
          } else {
            child.castShadow = true;
            child.receiveShadow = true;
          }

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
        if (child.isBone) {
          bonesMap.set(child.name, child);
          initialBoneRotations.set(child.name, child.rotation.clone());
        }
      });

      console.log(`Loaded ${targetModelType} model with ${allMeshes.size} modular parts, ${bonesMap.size} bones.`);

      // 3. Compute dynamic bounding box
      const box = new THREE.Box3().setFromObject(model);
      modelBounds.center = box.getCenter(new THREE.Vector3());
      modelBounds.size = box.getSize(new THREE.Vector3());
      modelBounds.maxDim = Math.max(modelBounds.size.x, modelBounds.size.y, modelBounds.size.z);

      camera.near = Math.max(0.01, modelBounds.maxDim * 0.01);
      camera.far = Math.max(100, modelBounds.maxDim * 30);
      camera.updateProjectionMatrix();

      // Adjust ground, shadow plane, and lighting scale
      const floorY = box.min.y;
      if (shadowPlane) {
        shadowPlane.position.y = floorY;
        shadowPlane.scale.set(modelBounds.maxDim / 2, modelBounds.maxDim / 2, 1);
      }
      if (grid) {
        grid.position.y = floorY + 0.001;
        grid.scale.set(modelBounds.maxDim / 2, 1, modelBounds.maxDim / 2);
      }
      if (dirLight) {
        dirLight.position.set(modelBounds.center.x + modelBounds.maxDim * 0.5, modelBounds.center.y + modelBounds.maxDim * 0.8, modelBounds.center.z + modelBounds.maxDim * 0.6);
        dirLight.shadow.camera.left = -modelBounds.maxDim;
        dirLight.shadow.camera.right = modelBounds.maxDim;
        dirLight.shadow.camera.top = modelBounds.maxDim * 1.5;
        dirLight.shadow.camera.bottom = -modelBounds.maxDim * 0.5;
        dirLight.shadow.camera.updateProjectionMatrix();
      }
      if (fillLight) fillLight.position.set(modelBounds.center.x - modelBounds.maxDim * 0.5, modelBounds.center.y + modelBounds.maxDim * 0.4, modelBounds.center.z - modelBounds.maxDim * 0.3);
      if (rimLight) rimLight.position.set(modelBounds.center.x, modelBounds.center.y + modelBounds.maxDim * 0.5, modelBounds.center.z - modelBounds.maxDim * 0.6);
      if (bottomLight) bottomLight.position.set(modelBounds.center.x, floorY - modelBounds.maxDim * 0.2, modelBounds.center.z + modelBounds.maxDim * 0.3);

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
    // 1. Hair (Supports all 6 hairstyles ported directly to Miku!)
    const mikuHairs = ['Hair_Miku_Twintails', 'Hair_Bob', 'Hair_Hime', 'Hair_Parted', 'Hair_Spiky', 'Hair_Twintails'];
    mikuHairs.forEach(h => {
      setMeshVisibility(h, h === cfg.hair);
    });

    // 2. Head & Facial Features
    setMeshVisibility('Head_Miku', true);
    setMeshVisibility('Face_Eyes_Left', true);
    setMeshVisibility('Face_Eyes_Right', true);
    setMeshVisibility('Face_Mouth_Miku', true);

    // 3. Outfits (Independent Top & Skirt)
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
    const hasSleeves = cfg.accessories.has('Outfit_Miku_Sleeves');
    setMeshVisibility('Outfit_Miku_Sleeves_Left', hasSleeves);
    setMeshVisibility('Outfit_Miku_Sleeves_Right', hasSleeves);

    // 5. Boots
    const hasBoots = (cfg.shoes === 'Shoes_Miku_Boots');
    setMeshVisibility('Shoes_Miku_Boots', hasBoots);

    // 6. Accessories
    setMeshVisibility('Accessory_Miku_Headset', cfg.accessories.has('Accessory_Miku_Headset'));
    setMeshVisibility('Accessory_Miku_Tie', cfg.accessories.has('Accessory_Miku_Tie'));
  } else if (currentLoadedModelType === 'mint') {
    // === MINT MODULAR PARTS ===
    // 1. Hair
    const hasBack = (cfg.hair === 'Hair_Mint_Full');
    const hasFront = (cfg.hair === 'Hair_Mint_Full' || cfg.hair === 'Hair_Mint_Front');
    setMeshVisibility('Hair_Mint_Back', hasBack);
    setMeshVisibility('Hair_Mint_Front', hasFront);

    // 2. Head & Facial Features
    setMeshVisibility('Head_Mint_Face', true);
    setMeshVisibility('Face_Mint_Eyes', true);
    setMeshVisibility('Face_Mint_Eyelashes', true);
    setMeshVisibility('Face_Mint_Highlight', true);
    setMeshVisibility('Face_Mint_Mask', true);

    // 3. Outfits
    // Note: In commercial anime game models like NTE Mint, the body skin and swimsuit are
    // unified into a single high-precision skinned mesh to prevent mesh clipping and optimize rendering.
    // Outfit_Mint_Swimsuit acts as her base body and is kept visible so she never turns into an invisible/floating head.
    const hasSkirt = (cfg.outfit === 'Outfit_Mint_Full' || cfg.outfit === 'Outfit_Mint_SkirtOnly');
    setMeshVisibility('Outfit_Mint_Swimsuit', true);
    setMeshVisibility('Outfit_Mint_Skirt', hasSkirt);

    // 4. Accessories
    setMeshVisibility('Accessory_Mint_Hat', cfg.accessories.has('Accessory_Mint_Hat'));
    setMeshVisibility('Accessory_Mint_ChestBow', cfg.accessories.has('Accessory_Mint_ChestBow'));
    setMeshVisibility('Accessory_Mint_Ribbon', cfg.accessories.has('Accessory_Mint_Ribbon'));
    setMeshVisibility('Outfit_Mint_Accessories', cfg.accessories.has('Outfit_Mint_Accessories'));
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
      } else if (mName.includes('Outfit') || mName.includes('Dress') || matName.includes('Outfit') || mName.includes('Accessory_Mint')) {
        mat.color = new THREE.Color(cfg.colors.outfit_primary);
        mat.needsUpdate = true;
      } else if (mName.includes('Body') || (mName.includes('Skin') && currentLoadedModelType !== 'miku' && currentLoadedModelType !== 'mint')) {
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
  const headBone = getBone('Head');
  if (headBone) {
    headBone.scale.set(p.head, p.head, p.head);
  }
}

// -----------------------------------------------------------------------------
// Unified Bone & Animation Control
// -----------------------------------------------------------------------------
function getBone(name) {
  return bonesMap.get(name) || bonesMap.get(`Bone_${name}`) || null;
}

function resetAllBones() {
  bonesMap.forEach((bone, name) => {
    const initRot = initialBoneRotations.get(name);
    if (initRot) {
      bone.rotation.copy(initRot);
    } else {
      bone.rotation.set(0, 0, 0);
    }
  });
}

function updateProceduralAnimations(delta) {
  if (!isPlaying || !model) return;

  animTime += delta * animSpeed;
  const t = animTime;

  const head = getBone('Head');
  const chest = getBone('Chest');
  const spine = getBone('Spine');
  const hips = getBone('Hips') || getBone('Center');

  const lArm = getBone('LeftUpperArm');
  const lFore = getBone('LeftLowerArm');
  const rArm = getBone('RightUpperArm');
  const rFore = getBone('RightLowerArm');

  const lLeg = getBone('LeftUpperLeg');
  const lKnee = getBone('LeftLowerLeg');
  const rLeg = getBone('RightUpperLeg');
  const rKnee = getBone('RightLowerLeg');

  // Miku twintails & skirt
  const lHair1 = getBone('LeftTwintail_1');
  const lHair2 = getBone('LeftTwintail_2');
  const rHair1 = getBone('RightTwintail_1');
  const rHair2 = getBone('RightTwintail_2');
  const skirtFL = getBone('Skirt_FrontLeft');
  const skirtFR = getBone('Skirt_FrontRight');
  const skirtB = getBone('Skirt_Back');

  if (currentAnimName === 'idle') {
    const breath = Math.sin(t * 2.2);
    if (chest) chest.rotation.x = breath * 0.08;
    if (spine) spine.rotation.x = breath * 0.04;
    if (head) {
      head.rotation.x = Math.sin(t * 1.5) * 0.06;
      head.rotation.y = Math.sin(t * 0.9) * 0.12;
    }
    if (lArm) lArm.rotation.z = 0.08 + Math.sin(t * 2.2) * 0.04;
    if (rArm) rArm.rotation.z = -0.08 - Math.sin(t * 2.2) * 0.04;

    // Twintails gentle swaying in breeze
    const hairSway = Math.sin(t * 2.0);
    if (lHair1) lHair1.rotation.z = 0.05 + hairSway * 0.08;
    if (lHair2) lHair2.rotation.z = 0.08 + hairSway * 0.06;
    if (rHair1) rHair1.rotation.z = -0.05 - hairSway * 0.08;
    if (rHair2) rHair2.rotation.z = -0.08 - hairSway * 0.06;
  }
  else if (currentAnimName === 'walk') {
    const cycle = t * 4.2;
    const sinCycle = Math.sin(cycle);
    const cosCycle = Math.cos(cycle);

    // Legs
    if (lLeg) lLeg.rotation.x = sinCycle * 0.55;
    if (rLeg) rLeg.rotation.x = -sinCycle * 0.55;
    if (lKnee) lKnee.rotation.x = Math.max(0, -sinCycle * 0.7);
    if (rKnee) rKnee.rotation.x = Math.max(0, sinCycle * 0.7);

    // Arms
    if (lArm) lArm.rotation.x = -sinCycle * 0.45;
    if (rArm) rArm.rotation.x = sinCycle * 0.45;

    // Torso sway
    if (spine) spine.rotation.y = sinCycle * 0.12;
    if (hips) hips.rotation.y = -sinCycle * 0.10;
    if (head) head.rotation.y = -sinCycle * 0.08;

    // Twintails bounce with stride
    const hairBounce = Math.abs(cosCycle) * 0.2;
    if (lHair1) {
      lHair1.rotation.x = -hairBounce;
      lHair1.rotation.z = 0.1 + sinCycle * 0.15;
    }
    if (rHair1) {
      rHair1.rotation.x = -hairBounce;
      rHair1.rotation.z = -0.1 + sinCycle * 0.15;
    }

    // Skirt motion
    if (skirtFL) skirtFL.rotation.x = Math.max(0, sinCycle * 0.25);
    if (skirtFR) skirtFR.rotation.x = Math.max(0, -sinCycle * 0.25);
  }
  else if (currentAnimName === 'dance') {
    const beat = t * 4.8;
    const sway = Math.sin(beat);

    if (hips) {
      hips.rotation.z = sway * 0.22;
      hips.rotation.y = Math.cos(beat * 0.5) * 0.25;
    }
    if (spine) spine.rotation.z = -sway * 0.18;
    if (chest) chest.rotation.y = sway * 0.25;
    if (head) head.rotation.z = sway * 0.18;

    if (rArm) {
      rArm.rotation.z = -0.6 + Math.sin(beat) * 0.45;
      rArm.rotation.x = Math.cos(beat) * 0.55;
    }
    if (lArm) {
      lArm.rotation.z = 0.6 - Math.sin(beat) * 0.45;
      lArm.rotation.x = -Math.cos(beat) * 0.55;
    }
    if (lFore) lFore.rotation.x = Math.abs(Math.sin(beat)) * 0.7;
    if (rFore) rFore.rotation.x = Math.abs(Math.cos(beat)) * 0.7;

    // Twintails energetic dance bounce
    if (lHair1) {
      lHair1.rotation.z = 0.15 + sway * 0.35;
      lHair1.rotation.x = Math.sin(beat * 2) * 0.25;
    }
    if (rHair1) {
      rHair1.rotation.z = -0.15 + sway * 0.35;
      rHair1.rotation.x = Math.sin(beat * 2) * 0.25;
    }
  }
  else if (currentAnimName === 'wave') {
    const wave = Math.sin(t * 7.5) * 0.45;
    if (rArm) {
      rArm.rotation.z = -2.1;
      rArm.rotation.y = 0.35;
      rArm.rotation.x = -0.20;
    }
    if (rFore) {
      rFore.rotation.y = 0.45;
      rFore.rotation.z = 0.35 + wave;
    }
    if (head) {
      head.rotation.y = -0.15;
      head.rotation.z = 0.12;
    }
    if (rHair1) rHair1.rotation.z = -0.15 + wave * 0.15;
  }
  else if (currentAnimName === 'salute') {
    if (rArm) {
      rArm.rotation.z = -1.15;
      rArm.rotation.y = -0.35;
      rArm.rotation.x = 0.20;
    }
    if (rFore) {
      rFore.rotation.z = -2.05;
      rFore.rotation.y = 0.35;
      rFore.rotation.x = -0.25;
    }
    if (chest) chest.rotation.x = -0.06;
    if (head) {
      head.rotation.x = 0.05;
      head.rotation.y = -0.05;
    }
  }
  else if (currentAnimName === 'bow') {
    const bowCycle = Math.sin(t * 1.8);
    const angle = Math.max(0, bowCycle) * 0.55;
    if (spine) spine.rotation.x = angle * 0.45;
    if (chest) chest.rotation.x = angle * 0.35;
    if (head) head.rotation.x = angle * 0.15;
    if (rArm) {
      rArm.rotation.x = -angle * 0.25;
      rArm.rotation.z = -0.10;
    }
    if (lArm) {
      lArm.rotation.x = -angle * 0.25;
      lArm.rotation.z = 0.10;
    }
    if (lHair1) lHair1.rotation.x = angle * 0.45;
    if (rHair1) rHair1.rotation.x = angle * 0.45;
  }
  else if (currentAnimName === 'pose') {
    if (head) {
      head.rotation.z = -0.15;
      head.rotation.y = 0.12;
      head.rotation.x = -0.05;
    }
    if (spine) spine.rotation.z = 0.08;
    if (hips) hips.rotation.z = -0.08;
    if (rArm) {
      rArm.rotation.z = -1.5;
      rArm.rotation.y = 0.5;
      rArm.rotation.x = 0.3;
    }
    if (rFore) rFore.rotation.z = -1.2;
    if (lArm) {
      lArm.rotation.z = 0.45;
      lArm.rotation.y = -0.2;
    }
    if (lFore) lFore.rotation.z = 0.8;
    if (rLeg) {
      rLeg.rotation.x = -0.15;
      rLeg.rotation.z = 0.08;
    }
    if (rKnee) rKnee.rotation.x = 0.3;
    if (lHair1) lHair1.rotation.z = 0.25;
    if (rHair1) rHair1.rotation.z = -0.2;
  }
}

function playAnimation(animName) {
  currentAnimName = animName;
  resetAllBones();
  animTime = 0;

  if (mixer) {
    const key = animName.toLowerCase();
    clipsMap.forEach((act) => act.stop());

    if (clipsMap.has(key)) {
      const action = clipsMap.get(key);
      action.reset().fadeIn(0.2).play();
    } else if (currentLoadedModelType === 'mint' && clipsMap.has('idle')) {
      const action = clipsMap.get('idle');
      action.reset().fadeIn(0.2).play();
    }
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
  const isMiku = (currentLoadedModelType === 'miku');
  const isMint = (currentLoadedModelType === 'mint');

  if (preset === 'full') {
    if (isMiku) {
      orbitControls.target.set(0, 10.0, 0);
      camera.position.set(0, 11.0, 27.0);
      orbitControls.minDistance = 2.0;
      orbitControls.maxDistance = 200.0;
    } else if (isMint) {
      orbitControls.target.set(0, 0.82, 0);
      camera.position.set(0, 0.90, 2.3);
      orbitControls.minDistance = 0.2;
      orbitControls.maxDistance = 20.0;
    } else {
      orbitControls.target.set(0, 0.05, 0);
      camera.position.set(0, 0.15, 2.7);
      orbitControls.minDistance = 0.2;
      orbitControls.maxDistance = 20.0;
    }
  } else if (preset === 'face') {
    if (isMiku) {
      orbitControls.target.set(0, 18.3, 0.5);
      camera.position.set(0, 18.3, 4.2);
      orbitControls.minDistance = 0.5;
      orbitControls.maxDistance = 100.0;
    } else if (isMint) {
      orbitControls.target.set(0, 1.47, 0.04);
      camera.position.set(0, 1.47, 0.45);
      orbitControls.minDistance = 0.05;
      orbitControls.maxDistance = 10.0;
    } else {
      orbitControls.target.set(0, 0.70, 0.05);
      camera.position.set(0, 0.70, 0.45);
      orbitControls.minDistance = 0.05;
      orbitControls.maxDistance = 10.0;
    }
  } else if (preset === 'torso') {
    if (isMiku) {
      orbitControls.target.set(0, 13.8, 0.2);
      camera.position.set(0, 14.0, 11.5);
      orbitControls.minDistance = 1.0;
      orbitControls.maxDistance = 100.0;
    } else if (isMint) {
      orbitControls.target.set(0, 1.15, 0.02);
      camera.position.set(0, 1.18, 1.05);
      orbitControls.minDistance = 0.1;
      orbitControls.maxDistance = 10.0;
    } else {
      orbitControls.target.set(0, 0.25, 0.02);
      camera.position.set(0, 0.28, 1.2);
      orbitControls.minDistance = 0.1;
      orbitControls.maxDistance = 10.0;
    }
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
      const clonedModel = SkeletonUtils.clone(model);

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
          const nameTag = (currentLoadedModelType === 'miku') ? 'miku_custom' : (currentLoadedModelType === 'mint' ? 'mint_custom' : `avatar_${avatarConfig.gender}`);
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
  const isMiku = (cfg.avatarModel === 'miku');
  const isMint = (cfg.avatarModel === 'mint');
  const isProcedural = (!isMiku && !isMint);

  // Filter cards by active model
  document.querySelectorAll('[data-model]').forEach((el) => {
    const models = el.dataset.model.split(',').map(s => s.trim());
    let show = false;
    if (isMiku && models.includes('miku')) show = true;
    if (isMint && models.includes('mint')) show = true;
    if (isProcedural && models.includes('procedural')) show = true;
    el.style.display = show ? '' : 'none';
  });

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
        avatarConfig.accessories.clear();
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
        } else if (val === 'mint') {
          avatarConfig.gender = 'female';
          avatarConfig.hair = 'Hair_Mint_Full';
          avatarConfig.outfit = 'Outfit_Mint_Full';
          avatarConfig.shoes = 'none';
          avatarConfig.accessories.add('Accessory_Mint_Hat');
          avatarConfig.accessories.add('Accessory_Mint_ChestBow');
          avatarConfig.accessories.add('Accessory_Mint_Ribbon');
          avatarConfig.accessories.add('Outfit_Mint_Accessories');
          avatarConfig.colors.hair = '#ffffff';
          avatarConfig.colors.outfit_primary = '#ffffff';
        } else {
          avatarConfig.gender = val;
          avatarConfig.hair = 'Hair_Twintails';
          avatarConfig.outfit = 'Outfit_Sailor';
          avatarConfig.shoes = 'Shoes_Loafers';
          avatarConfig.accessories.add('Accessory_CatEars');
          avatarConfig.colors.hair = '#33c7df';
          avatarConfig.colors.outfit_primary = '#1e293b';
        }
        syncUiToConfig();
        loadModularModel(val === 'miku' ? 'miku' : (val === 'mint' ? 'mint' : 'procedural'));
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
  if (currentLoadedModelType === 'mint') {
    const mintHairs = ['Hair_Mint_Full', 'Hair_Mint_Front', 'none'];
    const mintOutfits = ['Outfit_Mint_Full', 'Outfit_Mint_Bikini', 'Outfit_Mint_SkirtOnly', 'none'];
    const colors = ['#ffffff', '#ff85a2', '#ffd255', '#33c7df', '#a56de2', '#18181b'];
    avatarConfig.hair = mintHairs[Math.floor(Math.random() * mintHairs.length)];
    avatarConfig.outfit = mintOutfits[Math.floor(Math.random() * mintOutfits.length)];
    avatarConfig.colors.hair = colors[Math.floor(Math.random() * colors.length)];
    avatarConfig.colors.outfit_primary = colors[Math.floor(Math.random() * colors.length)];
    if (Math.random() > 0.5) avatarConfig.accessories.add('Accessory_Mint_Hat'); else avatarConfig.accessories.delete('Accessory_Mint_Hat');
    if (Math.random() > 0.5) avatarConfig.accessories.add('Accessory_Mint_ChestBow'); else avatarConfig.accessories.delete('Accessory_Mint_ChestBow');
    if (Math.random() > 0.5) avatarConfig.accessories.add('Accessory_Mint_Ribbon'); else avatarConfig.accessories.delete('Accessory_Mint_Ribbon');
    syncUiToConfig();
    applyAvatarConfiguration();
    return;
  }

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

  syncUiToConfig();
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
  if (isPlaying) {
    if (mixer && clipsMap.size > 0 && clipsMap.has(currentAnimName.toLowerCase())) {
      mixer.update(delta * animSpeed);
    } else {
      updateProceduralAnimations(delta);
    }
  }

  orbitControls.update();
  renderer.render(scene, camera);
}
