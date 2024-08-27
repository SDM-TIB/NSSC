from abc import ABC, abstractmethod

class AbstractUMLSearcher(ABC):
    """
    An abstract class representing a UMLS (Unified Medical Language System) searcher.

    This class provides the structure for implementing a UMLS searcher that aims to find the Concept Unique Identifier (CUI)
    for a given term. The subclasses should implement the search method to perform the actual search in UMLS.

    """

    @abstractmethod
    def search(self, term: str) -> str or None:
        """
        Performs a search in UMLS for the specified term and returns the corresponding CUI if found, or None if not found.

        Args:
            term (str): The term to search for in UMLS.

        Returns:
            str or None: The Concept Unique Identifier (CUI) if the term is found in UMLS, or None if the term is not found.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        pass


class Candidate:
    def __init__(self, cui, semantic_group, label):
        self.cui = cui
        self.semantic_group = semantic_group
        self.label = label
        self.definition = None

    def set_definition(self, definition):
        self.definition = definition

    def __str__(self):
        return f"CUI: {self.cui}, Semantic Group: {self.semantic_group}, Label: {self.label}, Definition: {self.definition}"

