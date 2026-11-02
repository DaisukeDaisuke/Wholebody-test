export const BODY_EDGES=[[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
export const FOOT_EDGES=[[15,17],[17,18],[18,19],[19,15],[16,20],[20,21],[21,22],[22,16]];
export const HAND_EDGES=[];
for(const start of[91,112])for(let f=0;f<5;f++){HAND_EDGES.push([start,start+1+4*f]);for(let j=0;j<3;j++)HAND_EDGES.push([start+1+4*f+j,start+2+4*f+j]);}
export const EDGES=[...BODY_EDGES,...FOOT_EDGES,...HAND_EDGES];
export const JOINT_NAMES=['鼻','左目','右目','左耳','右耳','左肩','右肩','左肘','右肘','左手首','右手首','左腰','右腰','左膝','右膝','左足首','右足首','左母趾先','左小趾先','左踵','右母趾先','右小趾先','右踵'];
export const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
export function boneLengths(heightM=1.70){return Object.fromEntries([[5,6,.23],[11,12,.17],[5,11,.30],[6,12,.30],[5,7,.186],[6,8,.186],[7,9,.146],[8,10,.146],[11,13,.245],[12,14,.245],[13,15,.246],[14,16,.246]].map(([a,b,f])=>[`${a}-${b}`,heightM*f]));}
