const REFINEMENT='https://huggingface.co/RabbitDaisuke/Wholebody-Refinement-onnx/resolve/main/';
export const REFINEMENT_MANIFEST=REFINEMENT+'models.json';
export const MODEL_SPECS = {
  'rtmw-l': {id:'rtmw-l',label:'RTMW-l · 全身133点／2視点向け',inputWH:[288,384],keypointCount:133,bytes:230000000,filename:'rtmw-l-384x288-fp32.onnx',url:REFINEMENT+'rtmw-l-384x288-fp32.onnx',archiveUrl:'https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip',license:'Apache-2.0',dimensions:2},
  'rtmpose-s': {id:'rtmpose-s',label:'RTMPose-s FP32 · 軽量比較',inputWH:[192,256],keypointCount:133,bytes:33559756,url:'https://huggingface.co/RabbitDaisuke/RTMPose-Wholebody-onnx/resolve/9d7f318c43452baf33f1b39d39c1f8c03512aeb5/rtmpose-s-256x192-fp32.onnx',sha256:'0dd72a9932d5f6eb539d02b3f5825e2ac5c15aeada5c0ff2dd9d5b34ab4103e8',license:'Apache-2.0',dimensions:2},
  'rtmw3d-x': {id:'rtmw3d-x',label:'RTMW3D-X · 単眼の相対3D＋距離推定',inputWH:[288,384],keypointCount:133,bytes:369330857,url:'https://huggingface.co/Soykaf/RTMW3D-x/resolve/a4f0d54c52ee44ee4e1dda1936d714490c6234d2/onnx/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx',sha256:'4a289c0e99d47eb595e99679d9d4a2d1def1b4241f9adcbafba44b9ff585ebcd',license:'Apache-2.0',dimensions:3,zRange:2.1744869},
  'hand5': {id:'hand5',label:'RTMPose Hand5 · 元映像から手指を再推論',inputWH:[256,256],keypointCount:21,bytes:55000000,filename:'rtmpose-hand5-256x256-fp32.onnx',url:REFINEMENT+'rtmpose-hand5-256x256-fp32.onnx',archiveUrl:'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip',license:'Apache-2.0',dimensions:2}
};
export const BODY_MODELS = ['rtmw-l','rtmw3d-x','rtmpose-s'].map(id=>MODEL_SPECS[id]);
