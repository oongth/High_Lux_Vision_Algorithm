// 하이라이트 압축 렌더링 (algorithm/highlight.py의 GPU 버전) + 변화 통계 합산
__device__ float sat(float x) { return fminf(1.f, fmaxf(0.f, x)); }
__device__ float smooth(float x) { x = sat(x); return x * x * (3 - 2 * x); }

// reflect 101
__device__ int mirror(int x, int n) {
    if (n <= 1) return 0;
    int p = 2 * n - 2;
    x = (x < 0 ? -x : x) % p;
    return x < n ? x : p - x;
}

__device__ float gray(const unsigned char* a, int x, int y, int w, int h) {
    int p = (mirror(y, h) * w + mirror(x, w)) * 3;
    return (a[p] * .114f + a[p + 1] * .587f + a[p + 2] * .299f) / 255.f;
}

extern "C" __global__ void highlight_render(
    const unsigned char* src, const float* fwd, const float* rev,
    unsigned char* dst, unsigned long long* stats,
    float* debug_y, float* debug_mask, float* debug_dark,
    int w, int height, float gamma, float reverse_gamma,
    float strength, float highlights, float shadows, float detail, float peak,
    float start, float severity, int view, int debug)
{
    int t = threadIdx.x, i = blockIdx.x * blockDim.x + t;
    __shared__ unsigned int sums[8][256];
    unsigned int v[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    if (i < w * height) {
        int p = 3 * i, x = i % w, ypos = i / w;
        const float weights[3] = {.114f, .587f, .299f};
        float o[3], d[3], y = 0.f, yf = 0.f, yd = 0.f, maximum = 0.f, minimum = 1.f;
        float fd = powf(fmaxf(fwd[i], 1e-6f), gamma);
        float rd = powf(fmaxf(rev[i], 1e-6f), reverse_gamma);

        // 후보, 휘도
        for (int c = 0; c < 3; c++) {
            o[c] = src[p + c] / 255.f;
            d[c] = 1.f - fminf(1.f, (1.f - o[c]) / rd);
            y += o[c] * weights[c];
            yf += fminf(1.f, o[c] / fd) * weights[c];
            yd += d[c] * weights[c];
            maximum = fmaxf(maximum, o[c]);
            minimum = fminf(minimum, o[c]);
        }

        // 목표 휘도
        float high = smooth((y - .48f) / .32f), low = smooth((.50f - y) / .35f);
        float reverse_gain = fminf(1.f, fmaxf(.65f - .30f * severity, yd / fmaxf(y, 1e-6f)));
        float lift = fminf(fmaxf(yf - y, 0.f), y * .60f);
        float dual = y + shadows * low * lift - .70f * highlights * high * y * (1 - reverse_gain);
        float a = ((1 - start) / (peak - start) - 1) * highlights;
        float target = dual > start ? start + (dual - start) / (1 + a * (dual - start) / (1 - start)) : dual;

        // 디테일
        const float k[5] = {1, 4, 6, 4, 1};
        float base = 0.f;
        for (int yy = -2; yy <= 2; yy++)
            for (int xx = -2; xx <= 2; xx++)
                base += k[yy + 2] * k[xx + 2] * gray(src, x + xx, ypos + yy, w, height) / 256.f;
        float texture = fminf(.03f, fmaxf(-.03f, y - base));
        float nonclipped = 1 - smooth((maximum - .94f) / .05f);
        target = sat(target + detail * highlights * high * nonclipped * texture);

        // 톤 확장
        float white_anchor = start + (1 - start) / (1 + a);
        float expanded = white_anchor * powf(fmaxf(target / white_anchor, 0.f), 1 + .75f * severity * highlights);
        target += high * (expanded - target);

        // 채도
        float chroma = 1 + .65f * highlights * high * fminf(2.5f, fmaxf(0.f, 1 / rd - 1));
        chroma = fminf(chroma, target / fmaxf(y - minimum, 1e-6f));
        chroma = fminf(chroma, (1 - target) / fmaxf(maximum - y, 1e-6f));

        // 출력, 통계
        float yr = 0.f;
        int difference = 0, largest = 0;
        bool white = true;
        for (int c = 0; c < 3; c++) {
            float mapped = target + chroma * (o[c] - y);
            float r = sat(o[c] + strength * (mapped - o[c]));
            int q = __float2int_rn(r * 255.f), s = src[p + c];
            float shown = view == 1 ? d[c] : (view == 2 ? sat(fabsf(r - o[c]) * 4) : r);
            dst[p + c] = (unsigned char)__float2int_rn(sat(shown) * 255.f);

            int delta = q - s;
            if (delta < 0) delta = -delta;
            difference += delta;
            largest = delta > largest ? delta : largest;
            yr += r * weights[c];
            white = white && (s >= 250);
            if (debug) debug_dark[p + c] = d[c];
        }

        bool bright = y >= .75f;
        int sy = __float2int_rn(y * 255.f), ry = __float2int_rn(yr * 255.f);
        v[0] = largest > 1;
        v[1] = difference;
        v[2] = bright;
        v[3] = bright ? sy : 0;
        v[4] = bright ? ry : 0;
        v[5] = white;
        v[6] = sy;
        v[7] = ry;
        if (debug) { debug_y[i] = target; debug_mask[i] = high; }
    }

    // 블록 합산
    for (int j = 0; j < 8; j++) sums[j][t] = v[j];
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (t < stride)
            for (int j = 0; j < 8; j++) sums[j][t] += sums[j][t + stride];
        __syncthreads();
    }
    if (t == 0)
        for (int j = 0; j < 8; j++) atomicAdd(stats + j, (unsigned long long)sums[j][0]);
}
