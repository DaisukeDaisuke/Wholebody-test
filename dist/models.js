const rt = {repo:'RabbitDaisuke/RTMPose-Wholebody-onnx',revision:'9d7f318c43452baf33f1b39d39c1f8c03512aeb5',width:192,height:256,dimensions:2,license:'Apache-2.0'};
const rw = {repo:'RabbitDaisuke/rtmw3d-webgpu-onnx',revision:'2fc6f257743eb890a340a50bbaa3e46c5a313e83',license:'Apache-2.0'};
export const BODY_MODELS = [
  {...rt,id:'rtmpose-s',name:'RTMPose-S · 軽量',path:'rtmpose-s-256x192-fp32.onnx',size:33559756,sha256:'0dd72a9932d5f6eb539d02b3f5825e2ac5c15aeada5c0ff2dd9d5b34ab4103e8'},
  {...rt,id:'rtmpose-t',name:'RTMPose-T · 最小',path:'rtmpose-t-256x192-fp32.onnx',size:22105541,sha256:'fcaae4a08ea7610b4bcc5d0b9e8a7e80fd6f2d998be41438a1b2737056a7623f'},
  {...rw,id:'rtmw-l',name:'RTMW-L 256 · 処理量を抑える',path:'rtmw2d/rtmw-l-256x192-fp32.onnx',width:192,height:256,dimensions:2,size:228801026,sha256:'8b6fb439dd02b02b0f71edaac647c8818e2335083ba1b3abc103f852a7c7c8d8'},
  {id:'rtmw-l-384',name:'RTMW-L 384 · 高解像度（既定）',repo:'RabbitDaisuke/Wholebody-Refinement-onnx',revision:'0a5519d19e52477307916e99ef22024ac0faa241',path:'rtmw-l-384x288-fp32.onnx',width:288,height:384,dimensions:2,size:229320930,sha256:'bd033156e5104c4f5d2edfe0453e02661e30a2f3da453ec93c8764d561b83054',license:'Apache-2.0'},
  {...rw,id:'rtmw3d-l',name:'RTMW3D-L · 単眼相対3D',path:'onnx/rtmw3d-l-fp32.onnx',width:288,height:384,dimensions:3,size:230006248,sha256:'4e37e5e9043817c5ff387b61c44bb48af94ad42e4978c4e43780761c65582abc'}
];
export const DETECTOR={name:'RTMDet-s',repo:'ziq/rtm',revision:'6bdd0110307167f5f003e92e34099a21bd61283c',path:'rtmdet-s.onnx',size:40309222,sha256:'3bb7246a8b5104dd94e9f9bbfcf92d2cd9e55d344a9a5e8b7b84ebcfeedafe14'};
export const HAND_MODEL={id:'mediapipe-hand',name:'MediaPipe 手専用 FP32',url:'https://media.githubusercontent.com/media/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/handpose_estimation_mediapipe/handpose_estimation_mediapipe_2023feb.onnx',size:4099621,sha256:'db0898ae717b76b075d9bf563af315b29562e11f8df5027a1ef07b02bef6d81c',license:'Apache-2.0'};
export const modelURL=m=>m.url||`https://huggingface.co/${m.repo}/resolve/${m.revision}/${m.path}`;
