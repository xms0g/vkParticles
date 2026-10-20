#pragma once

template<QueueType T>
void Device::submit(const float deltaTime, const float time) {
	const CommandBuffer* commandBuffer;
	uint64_t waitValue;
	uint64_t signalValue;
	vk::PipelineStageFlagBits2 stage;

	if constexpr (T == QueueType::Compute) {
		recordComputeCommandBuffer(deltaTime, time);

		commandBuffer = &mComputeCommandBuffers[mFrameIndex];
		waitValue = mComputeWaitValue;
		signalValue = mComputeSignalValue;
		stage = vk::PipelineStageFlagBits2::eComputeShader;
	} else {
		recordGraphicsCommandBuffer(mImageIndex, deltaTime, time);

		commandBuffer = &mGraphicsCommandBuffers[mFrameIndex];
		waitValue = mGraphicsWaitValue;
		signalValue = mGraphicsSignalValue;
		stage = vk::PipelineStageFlagBits2::eFragmentShader;
	}

	vk::SemaphoreSubmitInfo wait{
		.semaphore = *mSemaphore,
		.value = waitValue,
		.stageMask = stage
	};

	vk::CommandBufferSubmitInfo command{
		.commandBuffer = ***commandBuffer
	};

	vk::SemaphoreSubmitInfo signal{
		.semaphore = *mSemaphore,
		.value = signalValue,
		.stageMask = stage
	};

	const vk::SubmitInfo2 submitInfo{
		.waitSemaphoreInfoCount = 1,
		.pWaitSemaphoreInfos = &wait,

		.commandBufferInfoCount = 1,
		.pCommandBufferInfos = &command,

		.signalSemaphoreInfoCount = 1,
		.pSignalSemaphoreInfos = &signal
	};

	mQueue.submit2(submitInfo);
}

