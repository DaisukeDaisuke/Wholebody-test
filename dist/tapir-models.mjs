export const TAPIR_REPOSITORY='RabbitDaisuke/Foot-Point-Tracker-onnx';
// Set to the publication artifact's immutable revision after Actions parity.
export const TAPIR_MANIFEST_REVISION=null;
export const TAPIR_MANIFEST_URL=TAPIR_MANIFEST_REVISION?`https://huggingface.co/${TAPIR_REPOSITORY}/resolve/${TAPIR_MANIFEST_REVISION}/tapir-models.json`:null;
