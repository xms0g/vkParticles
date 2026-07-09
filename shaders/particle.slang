struct VSInput {
    float2 inPosition;
    float4 inColor;
};

struct VSOutput {
    float4 pos : SV_Position;
    float pointSize : SV_PointSize;
    float3 fragColor : COLOR0;
};

struct PSInput {
    float4 pos : SV_POSITION;
    float3 fragColor : COLOR0;
    float2 pointCoord : SV_PointCoord;
};

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput output;
    output.pointSize = 14.0;
    output.pos = float4(input.inPosition, 1.0, 1.0);
    output.fragColor = input.inColor.rgb;
    return output;
}

[shader("fragment")]
float4 fragMain(PSInput input) : SV_TARGET {
    float2 coord = input.pointCoord - float2(0.5);
    return float4(input.fragColor, 0.5 - length(coord));
}

struct PushConstants {
    float deltaTime;
};
[[vk::push_constant]] PushConstants pc;

struct Particle {
	float2 position;
	float2 velocity;
    float4 color;
};

StructuredBuffer<Particle> particlesIn;
RWStructuredBuffer<Particle> particlesOut;

[shader("compute")]
[numthreads(256,1,1)]
void compMain(uint3 threadId : SV_DispatchThreadID) {
    uint index = threadId.x;

    Particle p = particlesIn[index];
    const float2 gravity = float2(0.0, 0.5);
    // Integrate velocity
    p.velocity += gravity * pc.deltaTime;
    // Integrate position
    p.position += p.velocity * pc.deltaTime;

    // Bounce on X borders
    if (p.position.x <= -1.0) {
        p.position.x = -1.0;
        p.velocity.x *= -0.8;
    } else if (p.position.x >= 1.0) {
        p.position.x = 1.0;
        p.velocity.x *= -0.8;
    }

    // Bounce on Y borders
    if (p.position.y <= -1.0) {
        p.position.y = -1.0;
        p.velocity.y *= -0.8;
    } else if (p.position.y >= 1.0) {
        p.position.y = 1.0;
        p.velocity.y *= -0.8;
    }

    particlesOut[index] = p;
}