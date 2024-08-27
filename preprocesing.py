import json
import re
import json
from abc import ABC, abstractmethod

from textacy.preprocessing import normalize as normalize

class TextProcessor(ABC):
    @abstractmethod
    def convert(self, text:str)->str:
        pass

class Cleaner(TextProcessor):

    __REPLACEMENTS = [('`',""),('"',''),("\n  \n","\n")]

    def convert(self, text: str) -> str:

        # Eliminar espacios al principio o al final 
        text = text.strip()

        # Eliminar repeticiones de comas y espacios en blanco
        text = normalize.whitespace(text)
        text = normalize.repeating_chars(text, chars=",")
        text = normalize.repeating_chars(text, chars=":")
        text = normalize.repeating_chars(text, chars=";")

        # Eliminar caracteres no interesantes
        for old, new in self.__REPLACEMENTS:
            text = text.replace(old, new)

        return text


class PuntuactionFormater(TextProcessor):

    def convert(self, text: str) -> str:
         # Añadir espacio despues de "%" o "*" o "-" seguido de cadena
        text = re.sub(r"(%|\*|-)([a-z]+)", r"\1 \2", text) 

        # Añadir espacio tras cadena de texto seguido de signo de ";" o ":" 
        text = re.sub(r"([a-z])([:;])(\S)", r"\1\2 \3", text) 

        # Añadir espacio tras signo de "," o "."  seguido de cadena
        #self.__regex_show_test(text,r"(,|\.)([a-zA-Z]+)")
        text = re.sub(r"(,|\.)([a-zA-Z])", r"\1 \2", text) 
        
        # Añadir espacio entre palabras unidas por un "+""
        text = re.sub(r"([a-z])(\+)([a-z])", r"\1 \2 \3", text)

        # Añadir espacio entre "+"" seguido de una palabra
        text = re.sub(r"(\+)([a-z])", r"\1 \2", text)

        # Añadir espacio antes de cadena de texto seguido de (
        text = re.sub(r"([a-z])(\()", r"\1 \2", text)

        # Añadir espacio tras cadena de texto seguida de )
        text = re.sub(r"(\))([a-z])", r"\1 \2", text)
        
        # Añadir espacio entre guion y letra , cuando antes del guion hay un espacio
        text = re.sub(r"(\s)(-)([a-zA-Z])", r"\1\2 \3", text)

        # Añadir espacio entre guion y letra , cuando empieza la sentencia por guion
        text = re.sub(r"(\A-)([a-zA-Z])", r"\1 \2", text)

        return text
    
class Deacronimizer(TextProcessor):

    def __init__(self, acronimns) -> None:
        super().__init__()
        self.acronims = acronimns

    def convert(self, text: str) -> str:
        
        for old, new in self.acronims:
            regex = r"(\s|\(|\A)"+"("+old+")"+r"([\s),;:])"
            replace = r"\1"+ new +r"\3"
    
            #self.__regex_show_test(text, regex)
            text = re.sub(regex, replace, text)

        for old, new in self.acronims:
            regex = r"(\s|\(|\A)"+"("+old+")"+r"($)"
            replace = r"\1"+ new +r"\3"

            #self.__regex_show_test(text, regex)
            text = re.sub(regex, replace, text)

        for old, new in self.acronims:
            regex = r"(\s|\(|\A)"+"("+old+")"+r"(\.)"
            replace = r"\1"+ new 
    
            #self.__regex_show_test(text, regex)
            text = re.sub(regex, replace, text)

        return text


class Preprocesing:


    def __init__(self, path_acronims):
        self.acronims = self.__load_acronims(path_acronims)
        
    def fix(self, text):
   
        steps = [ Cleaner(), PuntuactionFormater(),
                 Deacronimizer(self.acronims)]
        
        for step in steps:
            text = step.convert(text)

        return text
  

    def __load_acronims(cls, path):
        with open(path) as f :
            dict_acro = json.load(f)

        return dict_acro["acronimos"]
    
