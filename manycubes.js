import * as THREE from 'three';
import { TrackballControls } from 'three/addons/controls/TrackballControls.js';
import Stats from 'Stats';

// Create the scene
const scene = new THREE.Scene();

// Create the camera
const camera = new THREE.PerspectiveCamera(
  75,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
camera.position.z = 5;

// Create the renderer
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Cube geometry
const geometry = new THREE.BoxGeometry(1, 1, 1);

// Generate random positions for cubes
const cubeCount = 50000;
const scale = 1000;
const cubePositions = [];
for (let i = 0; i < cubeCount; i++) {
  const x = Math.random() * scale - (scale/2);
  const y = Math.random() * scale - (scale/2);
  const z = Math.random() * scale - (scale/2);
  cubePositions.push(x, y, z);
}

// Create cubes and randomly place them
for (let i = 0; i < cubeCount; i++) {
  const material = new THREE.MeshPhongMaterial({ color: 0xffffff });
  const cube = new THREE.Mesh(geometry, material);
  const cubePosition = cubePositions.slice(i * 3, i * 3 + 3);
  cube.position.set(cubePosition[0], cubePosition[1], cubePosition[2]);
  scene.add(cube);
}

// Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.1);
//scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 0.2);
directionalLight.position.set(0, 1, 0);
//scene.add(directionalLight);

const spotLight = new THREE.SpotLight(0xFFFFFF, 1.0); 
spotLight.position.set(5, 0, 2);
spotLight.castShadow = true;
spotLight.shadow.mapSize.width = 10000;
spotLight.shadow.mapSize.height = spotLight.shadow.mapSize.width;
spotLight.penumbra = 1.0;
//scene.add(spotLight);

const lightBackGreen = new THREE.PointLight(0x00FF00, 1, 1000);
lightBackGreen.decay = 3.0;
lightBackGreen.position.set(5, 0, 2);
scene.add(lightBackGreen);

const lightBackWhite = new THREE.PointLight(0xFFFFFF, 5, 1000);
lightBackWhite.decay = 20.0;
lightBackWhite.position.set(5, 0, 2);
scene.add(lightBackWhite);

const stats = new Stats();
stats.showPanel(0);
document.getElementById("stats").appendChild(stats.dom);

const controls = new TrackballControls(camera, renderer.domElement);
controls.rotateSpeed = 1.0;
controls.zoomSpeed = 1.2;
controls.panSpeed = 0.8;
controls.noZoom = false;
controls.noPan = false;
controls.staticMoving = true;
controls.dynamicDampingFactor = 0.3;

// Animation loop
function animate() {
  stats.begin();

  requestAnimationFrame(animate);

  // Rotate the camera around the scene
  //camera.position.x = Math.sin(Date.now() * 0.0001) * 5;
  //camera.position.z = Math.cos(Date.now() * 0.0001) * 5;
  // Rotate lights, too
  
  controls.update();
  lightBackGreen.position.x = camera.position.x;
  lightBackGreen.position.y = camera.position.y;
  lightBackGreen.position.z = camera.position.z;
  lightBackWhite.position.x = camera.position.x;
  lightBackWhite.position.y = camera.position.y;
  lightBackWhite.position.z = camera.position.z;
  
  lightBackGreen.decay = 3.0 + 0.1 * Math.abs(Math.sin(Date.now() * 0.001));

  camera.lookAt(scene.position);
  renderer.render(scene, camera);

  stats.end();
}

animate();