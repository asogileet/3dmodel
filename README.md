# Gloria 3D 骨骼操作平台 (3D Rigging & Animation Studio)

本專案為目錄下的 `gloria.glb`（約 92MB、118 萬面人形模型）完成了自動骨架綁定與蒙皮權重計算，並打造了一個具備現代化 UI、流暢 3D 互動操作、動作預設庫、骨骼微調與光影設定的網頁應用程式。

---

## 🌟 主要功能特點

### 1. 骨架與蒙皮系統（Rigging & Skinning）
- **17 個標準人體關節骨骼**：
  - 核心幹部：骨盆（Hips）、脊椎（Spine）、胸腔（Chest）、頸部（Neck）、頭部（Head）
  - 上肢：左右上臂（UpperArm）、左右前臂/肘（LowerArm）、左右手（Hand）
  - 下肢：左右大腿（UpperLeg）、左右小腿/膝蓋（LowerLeg）、左右腳（Foot）
- **精確蒙皮權重（Skinning Weights）**：
  - 採用骨骼線段距離（Capsule Distance）平滑衰減，並施加左右側防穿透與高度空間保護。
  - 符合標準 glTF 2.0 規範（含 `JOINTS_0`、`WEIGHTS_0` 與世界逆綁定矩陣 `inverseBindMatrices`）。
  - 輸出產物：`gloria_rigged.glb`（支援 Blender、Three.js、Babylon.js、Godot 等主流引擎）。

### 2. 互動式 3D 網頁操作平台
- 🎮 **3D 關節操作軸（Gizmo）**：在 3D 視窗中直接點擊骨骼關節球，即可呼叫 3D 旋轉環以滑鼠自由旋轉關節。
- 🎚️ **FK 旋轉滑桿面板**：選擇任意關節，支援 X（俯仰）、Y（偏航）、Z（翻滾）精準角度微調，並可一鍵重設單一關節或全身 T-Pose。
- 💃 **動作預設庫（Preset Animations）**：
  - 🌬️ 待機呼吸（Idle Breathing）
  - 🚶 踏步前行（Walk Cycle）
  - 💃 活力律動跳舞（Dance Groove）
  - 👋 熱情揮手招呼（Wave Hand）
  - 🫡 敬禮致敬（Salute）
  - 🙇 鞠躬致謝（Bow）
  - 支援播放/暫停、0.2x ~ 2.0x 播放速度調節。
- 💡 **材質與光影調節**：
  - 材質模式：標準貼圖、線框模式（Wireframe）、純色黏土（Clay）、法線色彩（Normal）。
  - 主光源強度、環境光強度、地板陰影與格線開關。
  - 多種深淺背景色彩切換。
- 📷 **一鍵截圖**：下載當前擺放姿勢之高畫質無雜訊 PNG 圖片。

---

## 🚀 快速啟動方法

在專案目錄下執行以下指令即可啟動本地伺服器並自動開啟瀏覽器：

```bash
python start_server.py
```

瀏覽器將自動開啟：[http://localhost:8080/index.html](http://localhost:8080/index.html)

> 提示：若 8080 連接埠已被佔用，腳本會自動遞增至可用連接埠並顯示在終端機中。

---

## 📁 檔案架構

```text
3dmodel/
├── gloria.glb            # 原始 3D 模型 (~92 MB)
├── gloria_rigged.glb     # 綁定骨架後的 3D 模型 (~110 MB)
├── rig_model.py          # 自動生成骨骼與計算蒙皮權重的 Python 引擎
├── index.html            # 3D 模型操作平台網頁介面
├── style.css             # 質感深色毛玻璃風格 CSS
├── app.js                # Three.js 渲染、3D Gizmo、骨骼與動作控制器
├── start_server.py       # 本地伺服器啟動腳本（支援 CORS 與 GLB MIME）
└── README.md             # 專案說明文件
```

---

## 🛠️ 技術說明

- **前端渲染**：原生 ES Modules + [Three.js r160](https://threejs.org/)
- **控制器**：`OrbitControls`（相機視角旋轉縮放）+ `TransformControls`（骨骼 3D 旋轉軸）
- **骨骼視覺化**：`SkeletonHelper` + 自定義發光關節感應節點
- **無任何 npm 複雜建置**：單純靜態資源，載入即可運作。
