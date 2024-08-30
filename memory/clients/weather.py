import python_weather


async def get_weather(city: str):
    async with python_weather.Client(unit=python_weather.IMPERIAL) as client:
        return await client.get(city)
