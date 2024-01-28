#version 330 core
in vec4 colour;
in vec4 gl_FragCoord;   // but actually use UVs hey
in vec2 uv;
flat in uint fs_mode;

out vec4 frag_colour;

#define PI 3.1415926535897932384626433832795
#define ROOT2INV 0.70710678118654752440084436210484903928

uniform sampler2D tex;
uniform float time;

//so yea its the colouring

vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec4 acolour(float t) {
  return vec4(hsv2rgb(vec3(t, 1, 1)), 1);

  // if (t < 0.1) {
  //   return vec4(0., 1, 1, 1);
  // } else if (t < 0.5) {
  //   return vec4(0., 0., 0., 1);
  // } else if (t < 0.6) {
  //   return vec4(1., 0., 1., 1);
  // }
  // return vec4(0., 0., 0., 1.);
}

float rand(float n){return fract(sin(n) * 43758.5453123);}

float noise(float p){
	float fl = floor(p);
  float fc = fract(p);
	return mix(rand(fl), rand(fl + 1.0), fc);
}

float rand(vec2 n) { 
	return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
}

float noise(vec2 p){
	vec2 ip = floor(p);
	vec2 u = fract(p);
	u = u*u*(3.0-2.0*u);
	
	float res = mix(
		mix(rand(ip),rand(ip+vec2(1.0,0.0)),u.x),
		mix(rand(ip+vec2(0.0,1.0)),rand(ip+vec2(1.0,1.0)),u.x),u.y);
	return res*res;
}
float mod289(float x){return x - floor(x * (1.0 / 289.0)) * 289.0;}
vec4 mod289(vec4 x){return x - floor(x * (1.0 / 289.0)) * 289.0;}
vec4 perm(vec4 x){return mod289(((x * 34.0) + 1.0) * x);}

float noise(vec3 p){
    vec3 a = floor(p);
    vec3 d = p - a;
    d = d * d * (3.0 - 2.0 * d);

    vec4 b = a.xxyy + vec4(0.0, 1.0, 0.0, 1.0);
    vec4 k1 = perm(b.xyxy);
    vec4 k2 = perm(k1.xyxy + b.zzww);

    vec4 c = k2 + a.zzzz;
    vec4 k3 = perm(c);
    vec4 k4 = perm(c + 1.0);

    vec4 o1 = fract(k3 * (1.0 / 41.0));
    vec4 o2 = fract(k4 * (1.0 / 41.0));

    vec4 o3 = o2 * d.z + o1 * (1.0 - d.z);
    vec2 o4 = o3.yw * d.x + o3.xz * (1.0 - d.x);

    return o4.y * d.y + o4.x * (1.0 - d.y);
}
//	Classic Perlin 3D Noise 
//	by Stefan Gustavson
//
vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}
vec3 fade(vec3 t) {return t*t*t*(t*(t*6.0-15.0)+10.0);}

float cnoise(vec3 P){
  vec3 Pi0 = floor(P); // Integer part for indexing
  vec3 Pi1 = Pi0 + vec3(1.0); // Integer part + 1
  Pi0 = mod(Pi0, 289.0);
  Pi1 = mod(Pi1, 289.0);
  vec3 Pf0 = fract(P); // Fractional part for interpolation
  vec3 Pf1 = Pf0 - vec3(1.0); // Fractional part - 1.0
  vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
  vec4 iy = vec4(Pi0.yy, Pi1.yy);
  vec4 iz0 = Pi0.zzzz;
  vec4 iz1 = Pi1.zzzz;

  vec4 ixy = permute(permute(ix) + iy);
  vec4 ixy0 = permute(ixy + iz0);
  vec4 ixy1 = permute(ixy + iz1);

  vec4 gx0 = ixy0 / 7.0;
  vec4 gy0 = fract(floor(gx0) / 7.0) - 0.5;
  gx0 = fract(gx0);
  vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
  vec4 sz0 = step(gz0, vec4(0.0));
  gx0 -= sz0 * (step(0.0, gx0) - 0.5);
  gy0 -= sz0 * (step(0.0, gy0) - 0.5);

  vec4 gx1 = ixy1 / 7.0;
  vec4 gy1 = fract(floor(gx1) / 7.0) - 0.5;
  gx1 = fract(gx1);
  vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
  vec4 sz1 = step(gz1, vec4(0.0));
  gx1 -= sz1 * (step(0.0, gx1) - 0.5);
  gy1 -= sz1 * (step(0.0, gy1) - 0.5);

  vec3 g000 = vec3(gx0.x,gy0.x,gz0.x);
  vec3 g100 = vec3(gx0.y,gy0.y,gz0.y);
  vec3 g010 = vec3(gx0.z,gy0.z,gz0.z);
  vec3 g110 = vec3(gx0.w,gy0.w,gz0.w);
  vec3 g001 = vec3(gx1.x,gy1.x,gz1.x);
  vec3 g101 = vec3(gx1.y,gy1.y,gz1.y);
  vec3 g011 = vec3(gx1.z,gy1.z,gz1.z);
  vec3 g111 = vec3(gx1.w,gy1.w,gz1.w);

  vec4 norm0 = taylorInvSqrt(vec4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
  g000 *= norm0.x;
  g010 *= norm0.y;
  g100 *= norm0.z;
  g110 *= norm0.w;
  vec4 norm1 = taylorInvSqrt(vec4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
  g001 *= norm1.x;
  g011 *= norm1.y;
  g101 *= norm1.z;
  g111 *= norm1.w;

  float n000 = dot(g000, Pf0);
  float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
  float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
  float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
  float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
  float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
  float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
  float n111 = dot(g111, Pf1);

  vec3 fade_xyz = fade(Pf0);
  vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
  vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
  float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x); 
  return 2.2 * n_xyz;
}

float f1d(float p){
  return 
    1.000 * noise(p) +
    0.500 * noise(p*2.0 + 15123.34521) +
    0.250 * noise(p*4.0 + 13418.23523) +
    0.125 * noise(p*8.0 + 19023.52627) /
    1.875
    ;
}

// between 0 and 1
// then its -ln for mountain mode

float slowstart(float t) {
    return 1.0 - (1.0 - t)*(1.0 - t);
}
float slowstop(float t) {
    return t*t;
}

float quadraticInOut(float t) {
  float p = 2.0 * t * t;
  return t < 0.5 ? p : -p + (4.0 * t) - 1.0;
}
float exponentialInOut(float t) {
  return t == 0.0 || t == 1.0
    ? t
    : t < 0.5
      ? +0.5 * pow(2.0, (20.0 * t) - 10.0)
      : -0.5 * pow(2.0, 10.0 - (t * 20.0)) + 1.0;
}

void main() {
    switch(fs_mode) {
        case 0u:
        frag_colour = colour;
        break;
        case 1u:
        frag_colour = texture(tex, uv) * colour;
        break;
        default:
        frag_colour = vec4(0., 1., 0., 1.);
        break;
        case 2u:
        // 0.325
        float x = (0.5-uv.x);
        float y = (0.5-uv.y);
        float r = sqrt(x*x + y*y);
        float d_circle = abs(r - 0.325);
        float o = d_circle / (0.5 - 0.325);


        //float r = sqrt((0.5-uv.x)*(0.5-uv.x)+(0.5-uv.y)*(0.5-uv.y));
        // float mask = r < 0.15 ? 0.0 : r > 0.5 ? 0.0 : 1.0;
        
        float oo = (o*o);
        float oooo = (oo*oo);
        float mask = o > 1.0 ? 0.0 : 1.0 - oooo*oooo*oooo*oooo;
        float theta = atan(y, x);
        float theta1 = (theta + 0.66*PI*time);
        float theta2 = (theta + -0.5*PI*time);
        float t1 = mod(theta1, 2*PI);
        t1 -= PI;
        t1 = abs(t1) / PI;

        
        float t2 = mod(theta2, 2*PI);
        t2 -= PI;
        t2 = abs(t2) / PI;



        float inness = 1.0-o;
        // t1 *= inness;
        // t2 *= inness;
        
        float t = max(t1, t2);
        //float t = abs((max(mod(theta1, 2*PI), mod(theta2, 2*PI)) - PI) / (PI));
        t=t*t;
        t *= inness;
        frag_colour = mask*mix(colour, vec4(1., 1., 1., 1.), t);
        //frag_colour = mix(colour, vec4(0., 0., 0., 1.), exponentialInOut(t * 4.0));
        break;
        case 1000u:
        float h1 = 0.6 -0.3 * log(f1d(uv.x * 3 + time * 0.1));
      
        float h2 = 0.5 -0.2 * log(f1d(1238+(uv.x * 4) + (12238+time) * 0.2));
        float h3 = 0.4 -0.1 * log(f1d(7633+(uv.x * 5) + (55645+time) * 0.3));

        h1 = 1 - h1;
        h2 = 1 - h2;
        h3 = 1 - h3;

        // float h = 0.4 + 0.2 * f1d(uv.x * 10 + time * 1);
        if (uv.y > h3) {
          frag_colour = vec4(0.55, 0.39, 0.25, 1.0);
        } else if (uv.y > h2) {
          frag_colour = vec4(0.53, 0.33, 0.25, 1.0);
        } else if (uv.y > h1){
          frag_colour = vec4(0.5, 0.3, 0.25, 1.0);
        } else {
          frag_colour = vec4(0.55, 0.55, 0.9, 1.0);
        }
        break;
        case 1001u:
        // or if the geometry, it actually splits into 4 and they swap places, or flip and flip UVs
        // for transition diamonds shrink revealing next thing
        // if L1 dist > t - tlast kind of thing
        // the flag in crash team racing: with some kind of domain warping applied
        theta = PI/4;
        float up = cos(theta) * uv.x - sin(theta) * uv.y;
        float vp = sin(theta) * uv.x + cos(theta) * uv.y;

        up = up + time * 0.02;
        vp = vp + time * 0.0015;

        up *= 5.0;
        vp *= 5.0;

        up = mod(up, 1.0);
        vp = mod(vp, 1.0);

        if (up < 0.5 ^^ vp < 0.5) {
          frag_colour = vec4(0.6, 0.1, 0.6, 1.0);
        } else {
          frag_colour = vec4(0.3, 0.1, 0.6, 1.0);
        }

        break;
        case 1002u:
        float bw = 0.05;
        float tc = time * 0.8;
        float h = cnoise(vec3(10.0*uv.xy, 0.4 * time));
        float c1 = mod(tc, 1.0);
        float c2 = mod((tc + bw), 1.0);
        // theres something dumb about this that I pray I may understand tomorrow


        h = (h + 1.0) / 2.0;

        if (h > 0.5) {
          frag_colour = acolour(mod(h + tc, 1.0));
        } else {
          frag_colour = vec4(0., 0, 0, 1);
        }

        // if (c2 > c1) {
        //   if (h < c1) {
        //     frag_colour = vec4(0., 0., 0., 1.);
        //   } else if (h < c2) {
        //     frag_colour = vec4(0., 1., 1., 1.);
        //   } else {
        //     frag_colour = vec4(0., 0., 0., 1.);
        //   }
        // } else {
        //   if (h < c2) {
        //     frag_colour = vec4(0., 1., 1., 1.);
        //   } else if (h > c1) {
        //     frag_colour = vec4(0., 1., 1., 1.);
        //   } else {
        //     frag_colour = vec4(0., 0., 0., 1.);
        //   }
        // } 
        break;
    }
}

