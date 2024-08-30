from abc import ABC, abstractmethod


class Place(ABC):

    def __init__(self,
                 coordinates=None,
                 description=None):
        self.coordinates = coordinates
        self.description = description

    @abstractmethod
    def get_description(self):
        return self.description

    @abstractmethod
    def get_coordinates(self):
        # change coordinates
        return self.coordinates


class Country(Place):

    def __init__(self, name):
        super(Country, self).__init__()

        self.name = name

    def get_coordinates(self):
        pass

    def get_description(self):
        pass


class City(Place):

    def get_coordinates(self):
        pass

    def get_description(self):
        pass


class School(Place):
    def __init__(self,
                 city: City = None,
                 number: int = None,
                 ):
        super(School, self).__init__()
        self.city = city
        self.number = number

    def get_coordinates(self):
        pass

    def get_description(self):
        pass


class Home(Place):
    def __init__(self,
                 home_type: str = None,
                 city: City = None,
                 country: Country = None):
        super(Home, self).__init__()
        self.home_type = home_type
        self.city = city
        self.country = country

    def get_coordinates(self):
        pass

    def get_description(self):
        pass
