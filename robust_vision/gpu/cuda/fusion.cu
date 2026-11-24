// paper 모드 GPU 융합: 후보 생성, 품질 가중치, 피라미드 축소·확대, 누적
// 경계 밖 좌표를 반사(reflect 101) 방식으로 안쪽 좌표로 바꿈
__device__ int mirror(int x, int n) {
    if (n <= 1) return 0;
    int period = 2 * n - 2;
    x = (x < 0 ? -x : x) % period;
    return x < n ? x : period - x;
}

__device__ float gray(const float* a, int x, int y, int w, int h) {
    int p = (mirror(y, h) * w + mirror(x, w)) * 3;
    return a[p] * .114f + a[p + 1] * .587f + a[p + 2] * .299f;
}

// Eq. (7): contrast * saturation * exposedness (CPU의 _visual_quality와 같은 식)
__device__ float quality(const float* a, int x, int y, int w, int h) {
    int p = (y * w + x) * 3;
    float b = a[p], g = a[p + 1], r = a[p + 2];
    float mean = (b + g + r) / 3.f;

    float c = fabsf(gray(a, x - 1, y, w, h) + gray(a, x + 1, y, w, h)
                  + gray(a, x, y - 1, w, h) + gray(a, x, y + 1, w, h)
                  - 4 * gray(a, x, y, w, h));
    float s = sqrtf(((b - mean) * (b - mean) + (g - mean) * (g - mean) + (r - mean) * (r - mean)) / 3.f);
    float e = expf(-((b - .5f) * (b - .5f) + (g - .5f) * (g - .5f) + (r - .5f) * (r - .5f)) / .08f);
    return fmaxf(c, 1e-6f) * fmaxf(s, 1e-6f) * e;
}

// 원본, 밝게 보정한 후보, 어둡게 보정한 후보 생성
extern "C" __global__ void candidates(
    const unsigned char* src, const float* forward, const float* reverse,
    float* original, float* bright, float* dark, int count, float gamma)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count * 3) return;

    float v = src[i] / 255.f;
    original[i] = v;
    bright[i] = fminf(1.f, v / powf(fmaxf(forward[i / 3], 1e-6f), gamma));
    dark[i] = 1.f - fminf(1.f, (1.f - v) / powf(fmaxf(reverse[i / 3], 1e-6f), gamma));
}

// hard: 품질이 가장 높은 후보에 1 (동률이면 앞 후보), soft: 정규화한 품질
extern "C" __global__ void weights(
    const float* a, const float* b, const float* c,
    float* wa, float* wb, float* wc, int w, int h, int hard)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w * h) return;

    int x = i % w, y = i / w;
    float qa = quality(a, x, y, w, h);
    float qb = quality(b, x, y, w, h);
    float qc = quality(c, x, y, w, h);

    if (hard) {
        int winner = 0;
        float best = qa;
        if (qb > best) { winner = 1; best = qb; }
        if (qc > best) winner = 2;
        wa[i] = (winner == 0);
        wb[i] = (winner == 1);
        wc[i] = (winner == 2);
    } else {
        qa += 1e-12f; qb += 1e-12f; qc += 1e-12f;
        float sum = qa + qb + qc;
        wa[i] = qa / sum;
        wb[i] = qb / sum;
        wc[i] = qc / sum;
    }
}

// 5x5 가우시안으로 1/2 축소 (cv2.pyrDown과 같은 커널)
extern "C" __global__ void down(
    const float* src, float* dst, int w, int h, int ow, int oh, int channels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ow * oh * channels) return;
    int channel = i % channels;
    int x = (i / channels) % ow;
    int y = i / (channels * ow);

    const float k[5] = {1, 4, 6, 4, 1};
    float sum = 0;
    for (int dy = -2; dy <= 2; dy++)
        for (int dx = -2; dx <= 2; dx++)
            sum += k[dy + 2] * k[dx + 2]
                 * src[(mirror(2 * y + dy, h) * w + mirror(2 * x + dx, w)) * channels + channel];
    dst[i] = sum / 256.f;
}

// 2배 확대 (cv2.pyrUp과 같은 커널)
extern "C" __global__ void up(
    const float* src, float* dst, int w, int h, int ow, int oh, int channels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ow * oh * channels) return;
    int channel = i % channels;
    int x = (i / channels) % ow;
    int y = i / (channels * ow);

    const float k[5] = {1, 4, 6, 4, 1};
    float sum = 0;
    for (int dy = -2; dy <= 2; dy++) {
        int sy = mirror(y + dy, 2 * h);
        if (sy % 2) continue;
        for (int dx = -2; dx <= 2; dx++) {
            int sx = mirror(x + dx, 2 * w);
            if (sx % 2) continue;
            sum += k[dy + 2] * k[dx + 2] * src[((sy / 2) * w + sx / 2) * channels + channel];
        }
    }
    dst[i] = sum / 64.f;
}

// down/up을 가로(x)와 세로(y) 두 번으로 나눈 버전 (큰 영상에서 더 빠름)
extern "C" __global__ void down_x(
    const float* src, float* dst, int w, int h, int ow, int channels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ow * h * channels) return;
    int channel = i % channels;
    int x = (i / channels) % ow;
    int y = i / (channels * ow);

    const float k[5] = {1, 4, 6, 4, 1};
    float sum = 0;
    for (int dx = -2; dx <= 2; dx++)
        sum += k[dx + 2] * src[(y * w + mirror(2 * x + dx, w)) * channels + channel];
    dst[i] = sum / 16.f;
}

extern "C" __global__ void down_y(
    const float* src, float* dst, int w, int h, int oh, int channels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w * oh * channels) return;
    int channel = i % channels;
    int x = (i / channels) % w;
    int y = i / (channels * w);

    const float k[5] = {1, 4, 6, 4, 1};
    float sum = 0;
    for (int dy = -2; dy <= 2; dy++)
        sum += k[dy + 2] * src[(mirror(2 * y + dy, h) * w + x) * channels + channel];
    dst[i] = sum / 16.f;
}

extern "C" __global__ void up_x(
    const float* src, float* dst, int w, int h, int ow, int channels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ow * h * channels) return;
    int channel = i % channels;
    int x = (i / channels) % ow;
    int y = i / (channels * ow);

    const float k[5] = {1, 4, 6, 4, 1};
    float sum = 0;
    for (int dx = -2; dx <= 2; dx++) {
        int sx = mirror(x + dx, 2 * w);
        if (sx % 2) continue;
        sum += k[dx + 2] * src[(y * w + sx / 2) * channels + channel];
    }
    dst[i] = sum / 8.f;
}

extern "C" __global__ void up_y(
    const float* src, float* dst, int w, int h, int oh, int channels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w * oh * channels) return;
    int channel = i % channels;
    int x = (i / channels) % w;
    int y = i / (channels * w);

    const float k[5] = {1, 4, 6, 4, 1};
    float sum = 0;
    for (int dy = -2; dy <= 2; dy++) {
        int sy = mirror(y + dy, 2 * h);
        if (sy % 2) continue;
        sum += k[dy + 2] * src[((sy / 2) * w + x) * channels + channel];
    }
    dst[i] = sum / 8.f;
}

// Laplacian 레벨(level - expanded)에 가중치를 곱해 누적. 첫 후보면 덮어씀
extern "C" __global__ void accumulate(
    const float* level, const float* expanded, const float* weight,
    float* acc, int count, int last, int first)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    float value = (level[i] - (last ? 0.f : expanded[i])) * weight[i / 3];
    acc[i] = (first ? value : acc[i] + value);
}

extern "C" __global__ void add(const float* a, const float* b, float* out, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = a[i] + b[i];
}

// 0~1 float를 0~255 uint8로 반올림 변환
extern "C" __global__ void quantize(const float* src, unsigned char* dst, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = (unsigned char)__float2int_rn(fminf(1.f, fmaxf(0.f, src[i])) * 255.f);
}
