struct VSInput {
    float2 inPosition;
    float2 inVelocity;
};

struct VSOutput {
    float4 position : SV_Position;
    float pointSize : SV_PointSize;
    float3 fragColor : COLOR0;
};

struct PSInput {
    float3 fragColor : COLOR0;
};

static const float POINT_SIZE = 1.0;

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput output;
    output.pointSize = POINT_SIZE;
    output.position = float4(input.inPosition.x / pc.aspect, input.inPosition.y, 0.0, 1.0);
    float speed = clamp(length(input.inVelocity) * 0.7, 0.0, 1.0);
    output.fragColor = lerp(float3(0.08, 0.12, 0.35), float3(1.0, 0.5, 0.12), speed) * clamp(120000/ (uint)pc.count, 0.25, 4.0) * 0.5;
    return output;
}

[shader("fragment")]
float4 fragMain(PSInput input) : SV_TARGET {
    return float4(input.fragColor, 1.0);
}

struct PushConstants {
    float deltaTime;
    float time;
    float aspect;
    float count;
};
[[vk::push_constant]] PushConstants pc;

struct Particle {
	float2 position;
	float2 velocity;
};

[[vk::binding(0, 0)]] StructuredBuffer<Particle> particlesIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<Particle> particlesOut;

float hash(uint n) {
    uint x = n * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    x = (x >> 22u) ^ x;
    return (float)x / 4294967295.0;
}

[shader("compute")]
[numthreads(256,1,1)]
void compMain(uint3 threadId : SV_DispatchThreadID) {
    uint index = threadId.x;

    uint count;
    uint stride;
    particlesIn.GetDimensions(count, stride);

    if (index >= count) {
        return;
    }

    Particle p = particlesIn[index];
    float2 acc = float2(0.0);

    for (int k = 0; k < 3; k++) {
       float fk = (float)k;
       float a = pc.time * (0.25 + 0.08 * fk) + fk * 2.1;
       float2 c = float2(cos(a) * 1.1, sin(a * 1.3) * 0.55);
       float2 d = c - p.position;
       float r2 = dot(d, d) + 0.03;
       acc += d / r2 * 0.05 + float2(-d.y, d.x) / r2 * 0.09 * (k == 1 ? 1.0 : 1.0);
    }

    p.velocity = (p.velocity + acc * pc.deltaTime) * (1.0 - 0.35 * pc.deltaTime);
    float sp = length(p.velocity);
    if (sp > 2.5) {
        p.velocity = p.velocity * (2.5 / sp);
    }

    p.position += p.velocity * pc.deltaTime;
    if (abs(p.position.x) > pc.aspect * 1.05 || abs(p.position.y) > 1.05) {
        uint s = index * 3u + (uint)(pc.time * 60.0);
        p.position = float2((hash(s) * 2.0 - 1.0) * pc.aspect, hash(s + 1u) * 2.0 - 1.0);
        p.velocity = float2(0.0);
    }

    particlesOut[index] = p;
}