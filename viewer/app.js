import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';
import {STLLoader} from './vendor/STLLoader.js';

const host=document.querySelector('#scene'), status=document.querySelector('#status');
const scene=new THREE.Scene();scene.background=new THREE.Color('#101923');
const camera=new THREE.PerspectiveCamera(40,1,0.001,20);camera.up.set(0,0,1);
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));host.append(renderer.domElement);
const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
const reset=()=>{camera.position.set(.6,-.65,.5);controls.target.set(0,0,.18);controls.update();};reset();
document.querySelector('#reset').onclick=reset;
scene.add(new THREE.HemisphereLight(0xffffff,0x506070,2));const light=new THREE.DirectionalLight(0xffffff,3);light.position.set(1,-1,2);scene.add(light);
const grid=new THREE.GridHelper(1,20,0x617989,0x293c4a);grid.rotation.x=Math.PI/2;grid.position.z=-.002;scene.add(grid);
new ResizeObserver(()=>{camera.aspect=host.clientWidth/host.clientHeight;camera.updateProjectionMatrix();renderer.setSize(host.clientWidth,host.clientHeight);}).observe(host);
renderer.setAnimationLoop(()=>{controls.update();renderer.render(scene,camera);});
const mapping={shoulder_pan:'Rotation',shoulder_lift:'Pitch',elbow_flex:'Elbow',wrist_flex:'Wrist_Pitch',wrist_roll:'Wrist_Roll',gripper:'Jaw'};
const labels={shoulder_pan:'베이스',shoulder_lift:'어깨',elbow_flex:'팔꿈치',wrist_flex:'손목 굽힘',wrist_roll:'손목 회전',gripper:'그리퍼'};
const links={}, joints={};
const vector=(value,fallback='0 0 0')=>(value||fallback).trim().split(/\s+/).map(Number);
function origin(object,element){if(!element)return;object.position.fromArray(vector(element.getAttribute('xyz')));const [r,p,y]=vector(element.getAttribute('rpy'));object.quaternion.setFromEuler(new THREE.Euler(r,p,y,'ZYX'));}
try{
 const response=await fetch('/model/urdf/so101_new_calib.urdf');if(!response.ok)throw Error('모델 파일이 없습니다. python setup_viewer.py를 실행하세요.');
 const xml=new DOMParser().parseFromString(await response.text(),'application/xml');if(xml.querySelector('parsererror'))throw Error('Invalid URDF');
 const loader=new STLLoader(), pending=[];
 for(const link of xml.querySelectorAll('robot > link')){
  const group=new THREE.Group();links[link.getAttribute('name')]=group;
  for(const visual of link.querySelectorAll(':scope > visual')){
   const mesh=visual.querySelector('geometry > mesh');if(!mesh)continue;
   const frame=new THREE.Group();origin(frame,visual.querySelector('origin'));group.add(frame);
   const filename=mesh.getAttribute('filename').split('/').pop();
   const material=visual.querySelector('material');
   const namedMaterial=[...xml.querySelectorAll('robot > material')].find(item=>item.getAttribute('name')===material?.getAttribute('name'));
   const rgba=vector((material?.querySelector('color')||namedMaterial?.querySelector('color'))?.getAttribute('rgba'),'0.62 0.7 0.8 1');
   pending.push(loader.loadAsync('/model/meshes/'+filename).then(geometry=>{
    const obj=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color:new THREE.Color(...rgba.slice(0,3)),roughness:.65}));
    obj.scale.fromArray(vector(mesh.getAttribute('scale'),'1 1 1'));frame.add(obj);
   }));
  }
 }
 const children=new Set();
 for(const joint of xml.querySelectorAll('robot > joint')){
  const frame=new THREE.Group();origin(frame,joint.querySelector('origin'));
  const pivot=new THREE.Group();frame.add(pivot);const parent=joint.querySelector('parent').getAttribute('link'), child=joint.querySelector('child').getAttribute('link');
  links[parent].add(frame);pivot.add(links[child]);children.add(child);
  const limit=joint.querySelector('limit');joints[joint.getAttribute('name')]={pivot,axis:new THREE.Vector3(...vector(joint.querySelector('axis')?.getAttribute('xyz'),'1 0 0')).normalize(),type:joint.getAttribute('type'),low:Number(limit?.getAttribute('lower')??-Infinity),high:Number(limit?.getAttribute('upper')??Infinity)};
 }
 for(const [name,link]of Object.entries(links))if(!children.has(name))scene.add(link);
 await Promise.all(pending);status.textContent='로봇 데이터 대기 중';
 const poll=async()=>{
  try{
   const res=await fetch('/state',{cache:'no-store'});if(!res.ok)throw Error('연결 실패');const state=await res.json();
   const stale=!state.timestamp||Date.now()/1000-state.timestamp>1;
   status.textContent=stale?'업데이트 대기 / 정지 · '+state.status:state.status;status.style.color=stale?'#edc16e':'#71dbb3';
   document.querySelector('#source').textContent=state.source==='simulation'?'가상 데이터 · 실제 암 연결 없음':'실제 센서 데이터 · 도 / 그리퍼 %';
   for(const [motor,value]of Object.entries(state.positions)){
    const j=joints[mapping[motor]];if(!j)continue;
    // Same mapping as LeLab: degrees -> radians; gripper 0..100 -> Jaw.
    const angle=THREE.MathUtils.clamp(value*Math.PI/180,j.low,j.high);j.pivot.quaternion.setFromAxisAngle(j.axis,angle);
   }
   document.querySelector('#positions').replaceChildren(...Object.keys(mapping).map(key=>{const row=document.createElement('tr'),label=document.createElement('td'),value=document.createElement('td');label.textContent=labels[key];value.textContent=state.positions[key]===undefined?'—':state.positions[key].toFixed(1)+(key==='gripper'?' %':'°');row.append(label,value);return row;}));
  }catch(error){status.textContent='스크립트 연결 끊김';status.style.color='#ffb499';}
  setTimeout(poll,100);
 };poll();
}catch(error){document.querySelector('#error').textContent=error.message;status.textContent='모델 로딩 실패';console.error(error);}
