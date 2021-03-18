import asyncio

class DeviceSet:
    def __init__(self, num_devices):
        self.num_devices = num_devices
        self._free_devices = [i for i in range(num_devices)]
        self._semaphore = asyncio.Semaphore(num_devices)
        self._lock = asyncio.Lock()

    async def acquire_device(self):
        await self._semaphore.acquire()
        async with self._lock:
            return self._free_devices.pop()

    async def release_device(self, device_id):
        async with self._lock:
            self._free_devices.append(device_id)
        self._semaphore.release()
