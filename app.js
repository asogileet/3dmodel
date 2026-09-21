import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// Global App State
let scene, camera, renderer, orbitControls, transformControls;
let model, skeletonHelper, skinnedMeshes = [];
let bonesMap = new Map(); // name -> bone
let boneSpheres = [];     // clickable joint markers
let selectedBone = null;
let originalMaterials = new Map(); // mesh -> mat

// Current Loaded Model Meta
let currentModelFile = 'miku_psp_with_bones_fixed.glb';
let modelCenter = new THREE.Vector3(0, 10, 0);
let modelMaxDim = 20;

// Animation State
let isPlaying = true;
let currentAnim = 'idle';
let animSpeed = 1.0;
let animClock = new THREE.Clock();
let animTime = 0;
let initialBoneRotations = new Map(); // name -> initial Euler

// Performance & FPS State
let lastFpsTime = performance.now();
let frameCount = 0;

// Lighting & Environment
let dirLight, ambLight, fillLight, groundGrid, shadowPlane;

// DOM Elements
const canvas = document.getElementById('canvas3d');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingProgress = document.getElementById('loading-progress');
const loadingTitle = document.getElementById('loading-title');
const loadingDetail = document.getElementById('loading-detail');
const badge = document.getElementById('model-status-badge');
const modelSelect = document.getElementById('model-select');
const jointSelect = document.getElementById('joint-select');
const sliderRotX = document.getElementById('slider-rot-x');
const sliderRotY = document.getElementById('slider-rot-y');
const sliderRotZ = document.getElementById('slider-rot-z');
const valRotX = document.getElementById('val-rot-x');
const valRotY = document.getElementById('val-rot-y');
const valRotZ = document.getElementById('val-rot-z');
const fpsDisplay = document.getElementById('info-fps');

// Model Info DOM
const infoName = document.getElementById('info-name');
const infoFilesize = document.getElementById('info-filesize');
const infoMeshes = document.getElementById('info-meshes');
const infoBones = document.getElementById('info-bones');
const hierarchyPreview = document.getElementById('hierarchy-preview');

// Initialize
init();

function init() {
  // 1. Scene setup
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x12151f);

  // 2. Camera setup
  const aspect = canvas.clientWidth / canvas.clientHeight;
  camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 500);
  camera.position.set(0, 12, 28);

  // 3. Renderer setup
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
  renderer.setSize(canvas.clientWidth, canvas.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  // 4. OrbitControls
  orbitControls = new OrbitControls(camera, renderer.domElement);
  orbitControls.enableDamping = true;
  orbitControls.dampingFactor = 0.05;
  orbitControls.target.set(0, 10, 0);
  orbitControls.maxPolarAngle = Math.PI / 2 + 0.08;
  orbitControls.minDistance = 0.5;
  orbitControls.maxDistance = 200;

  // 5. TransformControls (Gizmo)
  transformControls = new TransformControls(camera, renderer.domElement);
  transformControls.setMode('rotate');
  transformControls.size = 0.8;
  scene.add(transformControls);

  transformControls.addEventListener('dragging-changed', (event) => {
    orbitControls.enabled = !event.value;
    if (event.value) {
      togglePlayPause(false);
    }
  });

  transformControls.addEventListener('change', () => {
    if (selectedBone) {
      updateSliderValuesFromBone(selectedBone);
    }
  });

  // 6. Lighting
  ambLight = new THREE.AmbientLight(0xffffff, 0.85);
  scene.add(ambLight);

  dirLight = new THREE.DirectionalLight(0xffffff, 1.3);
  dirLight.position.set(15, 30, 20);
  dirLight.castShadow = true;
  dirLight.shadow.mapSize.width = 2048;
  dirLight.shadow.mapSize.height = 2048;
  dirLight.shadow.camera.near = 0.5;
  dirLight.shadow.camera.far = 100;
  dirLight.shadow.bias = -0.0005;
  scene.add(dirLight);

  fillLight = new THREE.DirectionalLight(0x818cf8, 0.45);
  fillLight.position.set(-15, 15, -15);
  scene.add(fillLight);

  // 7. Ground & Grid
  const shadowGeo = new THREE.PlaneGeometry(100, 100);
  const shadowMat = new THREE.ShadowMaterial({ opacity: 0.35 });
  shadowPlane = new THREE.Mesh(shadowGeo, shadowMat);
  shadowPlane.rotation.x = -Math.PI / 2;
  shadowPlane.position.y = 0;
  shadowPlane.receiveShadow = true;
  scene.add(shadowPlane);

  groundGrid = new THREE.GridHelper(30, 30, 0x4f46e5, 0x222736);
  groundGrid.position.y = 0.001;
  scene.add(groundGrid);

  // 8. Event Listeners
  setupUIEventListeners();
  window.addEventListener('resize', onWindowResize);
  canvas.addEventListener('pointerdown', onCanvasPointerDown);

  // 9. Load Default Model (Miku)
  loadModel(currentModelFile);

  // 10. Start Loop
  animate();
}

function loadModel(modelUrl) {
  currentModelFile = modelUrl;

  // Cleanup existing model and helpers
  if (selectedBone) {
    transformControls.detach();
    selectedBone = null;
  }
  if (skeletonHelper) {
    scene.remove(skeletonHelper);
    skeletonHelper = null;
  }
  for (const { marker } of boneSpheres) {
    scene.remove(marker);
  }
  boneSpheres = [];
  if (model) {
    scene.remove(model);
    model = null;
  }

  bonesMap.clear();
  skinnedMeshes = [];
  originalMaterials.clear();
  initialBoneRotations.clear();

  loadingOverlay.style.display = 'flex';
  loadingOverlay.style.opacity = '1';
  loadingTitle.textContent = `正在載入 ${modelUrl === 'gloria_rigged.glb' ? 'Gloria' : '初音未來 (Miku)'}...`;
  loadingDetail.textContent = '讀取模型與骨架權重中...';
  loadingProgress.style.width = '0%';

  const loader = new GLTFLoader();
  const fetchUrl = modelUrl + (modelUrl.includes('?') ? '&' : '?') + 'v=2.2';

  loader.load(
    fetchUrl,
    (gltf) => {
      model = gltf.scene;
      scene.add(model);

      // Collect meshes and bones
      model.traverse((child) => {
        if (child.isMesh) {
          child.castShadow = true;
          child.receiveShadow = true;
          originalMaterials.set(child, child.material);
          if (child.isSkinnedMesh) {
            skinnedMeshes.push(child);
          }
        }
        if (child.isBone) {
          bonesMap.set(child.name, child);
          initialBoneRotations.set(child.name, child.rotation.clone());
        }
      });

      console.log(`Loaded ${modelUrl}: ${bonesMap.size} bones, ${skinnedMeshes.length} skinned meshes.`);

      // Compute bounding box & auto frame camera and environment
      const box = new THREE.Box3().setFromObject(model);
      modelCenter = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      modelMaxDim = Math.max(size.x, size.y, size.z);

      // Adjust camera
      orbitControls.target.copy(modelCenter);
      camera.position.set(modelCenter.x, modelCenter.y + modelMaxDim * 0.1, modelCenter.z + modelMaxDim * 1.4);
      camera.near = modelMaxDim * 0.01;
      camera.far = modelMaxDim * 25;
      camera.updateProjectionMatrix();
      orbitControls.update();

      // Adjust ground, grid, shadow plane
      const floorY = box.min.y;
      shadowPlane.position.y = floorY;
      groundGrid.position.y = floorY + 0.001;
      groundGrid.scale.set(modelMaxDim / 10, 1, modelMaxDim / 10);

      // Adjust lights
      dirLight.position.set(modelCenter.x + modelMaxDim * 0.8, modelCenter.y + modelMaxDim * 1.5, modelCenter.z + modelMaxDim * 1.2);
      dirLight.shadow.camera.left = -modelMaxDim;
      dirLight.shadow.camera.right = modelMaxDim;
      dirLight.shadow.camera.top = modelMaxDim * 1.5;
      dirLight.shadow.camera.bottom = -modelMaxDim * 0.5;
      dirLight.shadow.camera.updateProjectionMatrix();

      fillLight.position.set(modelCenter.x - modelMaxDim, modelCenter.y + modelMaxDim * 0.8, modelCenter.z - modelMaxDim);

      // SkeletonHelper
      skeletonHelper = new THREE.SkeletonHelper(model);
      skeletonHelper.material.linewidth = 2;
      skeletonHelper.material.color.set(0x818cf8);
      scene.add(skeletonHelper);

      // Joint Markers
      createJointMarkers(modelMaxDim);

      // Populate Joint Select Dropdown
      populateJointSelect();

      // Update Specs Info
      updateSpecsInfo(modelUrl, box);

      // Hide loading overlay
      loadingOverlay.style.opacity = '0';
      setTimeout(() => {
        loadingOverlay.style.display = 'none';
      }, 400);

      badge.textContent = '模型就緒';
      badge.style.color = '#10b981';
      badge.style.borderColor = 'rgba(16, 185, 129, 0.3)';

      // Auto select Head bone
      selectBoneByName('Head');
      if (!selectedBone) selectBoneByName('Bone_Head');
    },
    (xhr) => {
      if (xhr.lengthComputable) {
        const percent = Math.min(100, Math.round((xhr.loaded / xhr.total) * 100));
        loadingProgress.style.width = percent + '%';
        const mb = (xhr.loaded / (1024 * 1024)).toFixed(1);
        const totalMb = (xhr.total / (1024 * 1024)).toFixed(1);
        loadingDetail.textContent = `已下載 ${mb} MB / ${totalMb} MB (${percent}%)`;
      } else {
        const mb = (xhr.loaded / (1024 * 1024)).toFixed(1);
        loadingDetail.textContent = `已下載 ${mb} MB...`;
      }
    },
    (error) => {
      console.error('Error loading model:', error);
      loadingTitle.textContent = '模型載入失敗';
      loadingDetail.textContent = `無法載入 ${modelUrl}，請檢查檔案是否存在。`;
      badge.textContent = '載入錯誤';
      badge.style.color = '#ef4444';
    }
  );
}

function updateSpecsInfo(modelUrl, box) {
  const isMiku = modelUrl.includes('miku');
  if (infoName) infoName.textContent = isMiku ? '初音未來 (Hatsune Miku)' : 'Gloria';
  if (infoFilesize) infoFilesize.textContent = isMiku ? '~300 KB' : '~110 MB';
  if (infoMeshes) infoMeshes.textContent = `${skinnedMeshes.length} Skinned Meshes`;
  if (infoBones) infoBones.textContent = `${bonesMap.size} Joints`;

  if (hierarchyPreview) {
    if (isMiku) {
      hierarchyPreview.textContent = `Root\n└── Center (骨盆)\n    ├── Spine (脊椎)\n    │   └── Chest (胸腔)\n    │       ├── Neck -> Head (頭部)\n    │       │   ├── LeftTwintail (左雙馬尾 1..3)\n    │       │   └── RightTwintail (右雙馬尾 1..3)\n    │       ├── LeftShoulder -> Arm -> Hand (左手)\n    │       └── RightShoulder -> Arm -> Hand (右手)\n    └── Hips (下半身)\n        ├── Skirt (前後左右裙襬)\n        ├── LeftUpperLeg -> Knee -> Foot (左腿)\n        └── RightUpperLeg -> Knee -> Foot (右腿)`;
    } else {
      hierarchyPreview.textContent = `Root\n└── Bone_Hips (骨盆)\n    ├── Bone_Spine -> Bone_Chest (胸腔)\n    │   ├── Bone_Neck -> Bone_Head (頭部)\n    │   ├── Bone_LeftUpperArm -> Hand (左臂)\n    │   └── Bone_RightUpperArm -> Hand (右臂)\n    ├── Bone_LeftUpperLeg -> Foot (左腿)\n    └── Bone_RightUpperLeg -> Foot (右腿)`;
    }
  }
}

// Friendly Bone Names Dictionary
const friendlyNames = {
  // Miku standard names
  'Root': '根節點 (Root)',
  'Center': '身體中心 (Center)',
  'Spine': '脊椎 (Spine)',
  'Chest': '胸腔 (Chest)',
  'Neck': '頸部 (Neck)',
  'Head': '頭部 (Head)',
  'LeftEye': '左眼 (Left Eye)',
  'RightEye': '右眼 (Right Eye)',
  'LeftTwintail_1': '左雙馬尾根部 (Left Twintail 1)',
  'LeftTwintail_2': '左雙馬尾中段 (Left Twintail 2)',
  'LeftTwintail_3': '左雙馬尾末端 (Left Twintail 3)',
  'RightTwintail_1': '右雙馬尾根部 (Right Twintail 1)',
  'RightTwintail_2': '右雙馬尾中段 (Right Twintail 2)',
  'RightTwintail_3': '右雙馬尾末端 (Right Twintail 3)',
  'LeftShoulder': '左肩 (Left Shoulder)',
  'LeftUpperArm': '左上臂 (Left Upper Arm)',
  'LeftLowerArm': '左前臂 / 手肘 (Left Forearm)',
  'LeftHand': '左手腕 / 手掌 (Left Hand)',
  'RightShoulder': '右肩 (Right Shoulder)',
  'RightUpperArm': '右上臂 (Right Upper Arm)',
  'RightLowerArm': '右前臂 / 手肘 (Right Forearm)',
  'RightHand': '右手腕 / 手掌 (Right Hand)',
  'Hips': '骨盆 / 下半身 (Hips)',
  'Skirt_FrontLeft': '裙襬前左 (Skirt Front-L)',
  'Skirt_FrontRight': '裙襬前右 (Skirt Front-R)',
  'Skirt_Back': '裙襬後方 (Skirt Back)',
  'Skirt_SideLeft': '裙襬左側 (Skirt Side-L)',
  'Skirt_SideRight': '裙襬右側 (Skirt Side-R)',
  'LeftUpperLeg': '左大腿 (Left Thigh)',
  'LeftLowerLeg': '左小腿 / 膝蓋 (Left Knee)',
  'LeftFoot': '左腳踝 (Left Foot)',
  'LeftToe': '左腳尖 (Left Toe)',
  'RightUpperLeg': '右大腿 (Right Thigh)',
  'RightLowerLeg': '右小腿 / 膝蓋 (Right Knee)',
  'RightFoot': '右腳踝 (Right Foot)',
  'RightToe': '右腳尖 (Right Toe)',

  // Gloria names
  'Bone_Hips': '骨盆 (Hips)',
  'Bone_Spine': '脊椎 (Spine)',
  'Bone_Chest': '胸腔 (Chest)',
  'Bone_Neck': '頸部 (Neck)',
  'Bone_Head': '頭部 (Head)',
  'Bone_LeftUpperArm': '左上臂 (Left Upper Arm)',
  'Bone_LeftLowerArm': '左前臂 (Left Forearm)',
  'Bone_LeftHand': '左手 (Left Hand)',
  'Bone_RightUpperArm': '右上臂 (Right Upper Arm)',
  'Bone_RightLowerArm': '右前臂 (Right Forearm)',
  'Bone_RightHand': '右手 (Right Hand)',
  'Bone_LeftUpperLeg': '左大腿 (Left Thigh)',
  'Bone_LeftLowerLeg': '左小腿 (Left Knee)',
  'Bone_LeftFoot': '左腳 (Left Foot)',
  'Bone_RightUpperLeg': '右大腿 (Right Thigh)',
  'Bone_RightLowerLeg': '右小腿 (Right Knee)',
  'Bone_RightFoot': '右腳 (Right Foot)'
};

function populateJointSelect() {
  jointSelect.innerHTML = '<option value="">-- 請選擇關節 --</option>';

  bonesMap.forEach((bone, name) => {
    // Skip tiny end helper bones from cluttering the dropdown if desired
    if (name.includes('end') || name.includes('Thumb_') || name.includes('Index_') || name.includes('Middle_') || name.includes('Ring_') || name.includes('Pinky_')) {
      return;
    }
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = friendlyNames[name] || name;
    jointSelect.appendChild(opt);
  });
}

function createJointMarkers(scaleDim) {
  const radius = Math.max(0.012, scaleDim * 0.013);
  const sphereGeo = new THREE.SphereGeometry(radius, 14, 14);
  const sphereMat = new THREE.MeshStandardMaterial({
    color: 0x4f46e5,
    emissive: 0x312e81,
    roughness: 0.3,
    metalness: 0.8
  });

  bonesMap.forEach((bone, name) => {
    // Only display markers for main joints to avoid cluttering with 40 finger joints
    if (name.includes('end') || name.includes('Thumb_') || name.includes('Index_') || name.includes('Middle_') || name.includes('Ring_') || name.includes('Pinky_')) {
      return;
    }
    const marker = new THREE.Mesh(sphereGeo, sphereMat.clone());
    marker.userData = { boneName: name, isJointMarker: true };
    scene.add(marker);
    boneSpheres.push({ marker, bone });
  });
}

function updateJointMarkers() {
  const worldPos = new THREE.Vector3();
  for (const { marker, bone } of boneSpheres) {
    bone.getWorldPosition(worldPos);
    marker.position.copy(worldPos);
    if (selectedBone === bone) {
      marker.material.color.set(0xec4899); // Highlight selected
      marker.material.emissive.set(0xbe185d);
      marker.scale.set(1.4, 1.4, 1.4);
    } else {
      marker.material.color.set(0x4f46e5);
      marker.material.emissive.set(0x312e81);
      marker.scale.set(1.0, 1.0, 1.0);
    }
  }
}

function selectBoneByName(boneName) {
  // Support alias lookups
  let bone = bonesMap.get(boneName) || bonesMap.get(`Bone_${boneName}`);
  if (!bone) {
    // Try finding by ending substring
    for (const [key, b] of bonesMap.entries()) {
      if (key.endsWith(boneName) || key.toLowerCase().includes(boneName.toLowerCase())) {
        bone = b;
        boneName = key;
        break;
      }
    }
  }
  if (!bone) return;

  selectedBone = bone;
  jointSelect.value = bone.name;
  transformControls.attach(bone);
  updateSliderValuesFromBone(bone);
}

function updateSliderValuesFromBone(bone) {
  const degX = Math.round(THREE.MathUtils.radToDeg(bone.rotation.x));
  const degY = Math.round(THREE.MathUtils.radToDeg(bone.rotation.y));
  const degZ = Math.round(THREE.MathUtils.radToDeg(bone.rotation.z));

  sliderRotX.value = degX;
  sliderRotY.value = degY;
  sliderRotZ.value = degZ;

  valRotX.textContent = `${degX}°`;
  valRotY.textContent = `${degY}°`;
  valRotZ.textContent = `${degZ}°`;
}

function applySliderToBone() {
  if (!selectedBone) return;
  if (isPlaying) {
    togglePlayPause(false);
  }

  const radX = THREE.MathUtils.degToRad(parseFloat(sliderRotX.value));
  const radY = THREE.MathUtils.degToRad(parseFloat(sliderRotY.value));
  const radZ = THREE.MathUtils.degToRad(parseFloat(sliderRotZ.value));

  selectedBone.rotation.set(radX, radY, radZ);
  valRotX.textContent = `${sliderRotX.value}°`;
  valRotY.textContent = `${sliderRotY.value}°`;
  valRotZ.textContent = `${sliderRotZ.value}°`;
}

function resetCurrentBone() {
  if (!selectedBone) return;
  const initRot = initialBoneRotations.get(selectedBone.name);
  if (initRot) {
    selectedBone.rotation.copy(initRot);
  } else {
    selectedBone.rotation.set(0, 0, 0);
  }
  updateSliderValuesFromBone(selectedBone);
}

function resetAllBones(pause = true) {
  if (pause) {
    togglePlayPause(false);
  }
  bonesMap.forEach((bone, name) => {
    const initRot = initialBoneRotations.get(name);
    if (initRot) {
      bone.rotation.copy(initRot);
    } else {
      bone.rotation.set(0, 0, 0);
    }
  });
  if (selectedBone) {
    updateSliderValuesFromBone(selectedBone);
  }
}

// Unified Bone Finder
function getBone(name) {
  return bonesMap.get(name) || bonesMap.get(`Bone_${name}`) || null;
}

// Procedural Animation System (Supports both Miku and Gloria)
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

  if (currentAnim === 'idle') {
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
  else if (currentAnim === 'walk') {
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
  else if (currentAnim === 'dance') {
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
  else if (currentAnim === 'wave') {
    const wave = Math.sin(t * 7.5) * 0.55;
    if (rArm) {
      rArm.rotation.z = -1.8;
      rArm.rotation.x = 0.3;
    }
    if (rFore) {
      rFore.rotation.z = -0.4 + wave;
      rFore.rotation.x = 0.2;
    }
    if (head) {
      head.rotation.y = -0.25;
      head.rotation.z = 0.15;
    }
    if (rHair1) rHair1.rotation.z = -0.3 + wave * 0.2;
  }
  else if (currentAnim === 'salute') {
    if (rArm) {
      rArm.rotation.z = -1.3;
      rArm.rotation.x = 0.45;
      rArm.rotation.y = 0.5;
    }
    if (rFore) {
      rFore.rotation.x = 1.4;
      rFore.rotation.z = -0.45;
    }
    if (chest) chest.rotation.x = -0.08;
    if (head) head.rotation.x = 0.05;
  }
  else if (currentAnim === 'bow') {
    const bowCycle = Math.sin(t * 1.8);
    const angle = Math.max(0, bowCycle) * 0.6;
    if (hips) hips.rotation.x = angle * 0.5;
    if (spine) spine.rotation.x = angle * 0.6;
    if (chest) chest.rotation.x = angle * 0.4;
    if (head) head.rotation.x = angle * 0.3;
    if (rArm) rArm.rotation.x = angle * 0.4;
    if (lArm) lArm.rotation.x = angle * 0.4;
    if (lHair1) lHair1.rotation.x = angle * 0.5;
    if (rHair1) rHair1.rotation.x = angle * 0.5;
  }
}

function togglePlayPause(play) {
  if (play === undefined) isPlaying = !isPlaying;
  else isPlaying = play;

  const btnIcon = document.getElementById('play-pause-icon');
  if (isPlaying) {
    btnIcon.textContent = '⏸️';
  } else {
    btnIcon.textContent = '▶️';
  }
}

// Raycasting to click joint spheres
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

function onCanvasPointerDown(event) {
  if (event.button !== 0) return;

  const rect = canvas.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const clickableObjects = boneSpheres.map(b => b.marker);
  const intersects = raycaster.intersectObjects(clickableObjects);

  if (intersects.length > 0) {
    const hitMarker = intersects[0].object;
    const boneName = hitMarker.userData.boneName;
    selectBoneByName(boneName);
  }
}

// Setup All UI Event Handlers
function setupUIEventListeners() {
  // Model Select Dropdown
  if (modelSelect) {
    modelSelect.addEventListener('change', (e) => {
      loadModel(e.target.value);
    });
  }

  // Tab Switching
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetPane = document.getElementById(btn.dataset.tab);
      if (targetPane) targetPane.classList.add('active');
    });
  });

  // Joint Select Dropdown
  jointSelect.addEventListener('change', (e) => {
    if (e.target.value) {
      selectBoneByName(e.target.value);
    }
  });

  // FK Rotation Sliders
  [sliderRotX, sliderRotY, sliderRotZ].forEach(slider => {
    slider.addEventListener('input', applySliderToBone);
  });

  // Quick Bone Buttons
  document.querySelectorAll('[data-quick-bone]').forEach(btn => {
    btn.addEventListener('click', () => {
      selectBoneByName(btn.dataset.quickBone);
    });
  });

  // Reset Bone Buttons
  document.getElementById('btn-reset-current-bone').addEventListener('click', resetCurrentBone);
  document.getElementById('btn-reset-all-pose').addEventListener('click', () => resetAllBones(true));

  // Animation Play/Pause & Speed
  document.getElementById('btn-play-pause').addEventListener('click', () => togglePlayPause());
  document.getElementById('btn-stop-anim').addEventListener('click', () => {
    togglePlayPause(false);
    resetAllBones(true);
  });

  const speedSlider = document.getElementById('slider-anim-speed');
  const speedVal = document.getElementById('val-anim-speed');
  speedSlider.addEventListener('input', (e) => {
    animSpeed = parseFloat(e.target.value);
    speedVal.textContent = `${animSpeed.toFixed(1)}x`;
  });

  // Preset Animation Cards
  document.querySelectorAll('.anim-card').forEach(card => {
    card.addEventListener('click', () => {
      document.querySelectorAll('.anim-card').forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      currentAnim = card.dataset.anim;
      resetAllBones(false);
      togglePlayPause(true);
    });
  });

  // Viewport Toolbar Buttons
  const toggleSkelBtn = document.getElementById('btn-toggle-skeleton');
  toggleSkelBtn.addEventListener('click', () => {
    const isVisible = skeletonHelper ? skeletonHelper.visible : true;
    if (skeletonHelper) skeletonHelper.visible = !isVisible;
    boneSpheres.forEach(b => b.marker.visible = !isVisible);
    toggleSkelBtn.classList.toggle('active', !isVisible);
  });

  const toggleGizmoBtn = document.getElementById('btn-toggle-gizmo');
  toggleGizmoBtn.addEventListener('click', () => {
    const isGizmoEnabled = transformControls.enabled;
    transformControls.enabled = !isGizmoEnabled;
    transformControls.visible = !isGizmoEnabled;
    toggleGizmoBtn.classList.toggle('active', !isGizmoEnabled);
  });

  document.getElementById('btn-reset-view').addEventListener('click', () => {
    camera.position.set(modelCenter.x, modelCenter.y + modelMaxDim * 0.1, modelCenter.z + modelMaxDim * 1.4);
    orbitControls.target.copy(modelCenter);
    orbitControls.update();
  });

  document.getElementById('btn-screenshot').addEventListener('click', takeScreenshot);

  // Material & Shading
  document.getElementById('btn-mat-texture').addEventListener('click', () => setShadingMode('texture'));
  document.getElementById('btn-mat-wireframe').addEventListener('click', () => setShadingMode('wireframe'));
  document.getElementById('btn-mat-clay').addEventListener('click', () => setShadingMode('clay'));
  document.getElementById('btn-mat-normal').addEventListener('click', () => setShadingMode('normal'));

  // Lighting & Grid
  const dirLightSlider = document.getElementById('slider-light-dir');
  const dirLightVal = document.getElementById('val-light-dir');
  dirLightSlider.addEventListener('input', (e) => {
    dirLight.intensity = parseFloat(e.target.value);
    dirLightVal.textContent = dirLight.intensity.toFixed(1);
  });

  const ambLightSlider = document.getElementById('slider-light-amb');
  const ambLightVal = document.getElementById('val-light-amb');
  ambLightSlider.addEventListener('input', (e) => {
    ambLight.intensity = parseFloat(e.target.value);
    ambLightVal.textContent = ambLight.intensity.toFixed(1);
  });

  const shadowBtn = document.getElementById('btn-toggle-shadow');
  shadowBtn.addEventListener('click', () => {
    shadowPlane.visible = !shadowPlane.visible;
    shadowBtn.classList.toggle('active', shadowPlane.visible);
  });

  const gridBtn = document.getElementById('btn-toggle-grid');
  gridBtn.addEventListener('click', () => {
    groundGrid.visible = !groundGrid.visible;
    gridBtn.classList.toggle('active', groundGrid.visible);
  });

  // Background Colors
  document.querySelectorAll('[data-bg]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-bg]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      scene.background = new THREE.Color(btn.dataset.bg);
    });
  });
}

function setShadingMode(mode) {
  document.querySelectorAll('#tab-env .grid-2 .btn').forEach(b => b.classList.remove('active'));
  document.getElementById(`btn-mat-${mode}`).classList.add('active');

  const clayMat = new THREE.MeshStandardMaterial({ color: 0xd1d5db, roughness: 0.6, metalness: 0.1 });
  const normalMat = new THREE.MeshNormalMaterial();

  for (const mesh of skinnedMeshes) {
    if (mode === 'texture') {
      mesh.material = originalMaterials.get(mesh);
      mesh.material.wireframe = false;
    } else if (mode === 'wireframe') {
      mesh.material = originalMaterials.get(mesh);
      mesh.material.wireframe = true;
    } else if (mode === 'clay') {
      mesh.material = clayMat;
      mesh.material.wireframe = false;
    } else if (mode === 'normal') {
      mesh.material = normalMat;
      mesh.material.wireframe = false;
    }
    mesh.material.needsUpdate = true;
  }
}

function takeScreenshot() {
  const isSkelVis = skeletonHelper ? skeletonHelper.visible : false;
  const isGizmoVis = transformControls.visible;

  if (skeletonHelper) skeletonHelper.visible = false;
  transformControls.visible = false;
  boneSpheres.forEach(b => b.marker.visible = false);

  renderer.render(scene, camera);
  const dataUrl = renderer.domElement.toDataURL('image/png');

  if (skeletonHelper) skeletonHelper.visible = isSkelVis;
  transformControls.visible = isGizmoVis;
  boneSpheres.forEach(b => b.marker.visible = isSkelVis);

  const link = document.createElement('a');
  link.download = `model_pose_${Date.now()}.png`;
  link.href = dataUrl;
  link.click();
}

function onWindowResize() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}

function animate() {
  requestAnimationFrame(animate);

  const delta = Math.min(animClock.getDelta(), 0.1);
  updateProceduralAnimations(delta);

  orbitControls.update();
  updateJointMarkers();

  renderer.render(scene, camera);

  frameCount++;
  const now = performance.now();
  if (now - lastFpsTime >= 1000) {
    if (fpsDisplay) {
      fpsDisplay.textContent = `${frameCount} FPS`;
    }
    frameCount = 0;
    lastFpsTime = now;
  }
}
