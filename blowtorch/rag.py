from pathlib import PosixPath, Path
from __init__ import client
import pdfplumber
import re

class Paragraph:

    id = 0

    def __init__(self, depth: str, heading: str, paragraph: str, parent: "Paragraph|None"=None) -> None:

        # assign new id
        self.id = Paragraph.id
        Paragraph.id += 1

        self.parent: Paragraph|None = parent
        self.children: set = set()
        
        self.depth = depth  # depth 1 = <h1> heading
        self.heading = heading
        self.paragraph = paragraph
        self.abstraction: list = [self.paragraph] # abstraction level, default i.e. full text is index 0
    
    def add_child (self, child: "Paragraph|None"=None) -> None:

        self.children.add(child)

class Doc:

    def __init__(self, file: str|PosixPath):

        # convert to posix
        self.file = Path(file) if not isinstance(file, PosixPath) else file

        # determine the text heights
        self.heights: set = self.extract_font_block_heights()
        self.height_map: dict[float, str] = dict()
        self.heights_sorted = list(self.heights)
        self.heights_sorted.reverse()

        # fill height map which will assign heading sizes
        ind = 1
        for h in self.heights_sorted:
            self.height_map[h] = ind
            ind += 1
        
        # extract paragraphs
        self.paragraphs = self.extract_paragraphs()
        self.title = self.paragraphs[0]

        # set tree-like relation between paragraphs
        for i in range(1, len(self.paragraphs)):
            p = self.paragraphs[i]
            p_prev = self.paragraphs[i-1]
            if p.depth > p_prev.depth:
                parent = p_prev
            elif p.depth == p_prev.depth:
                parent = p_prev.parent
            else:
                for j in range(i, -1, -1):
                    parent = self.paragraphs[j]
                    if parent.depth < p.depth:
                        break
            # set edge connection
            parent.add_child(p)
            p.parent = parent

        for p in self.paragraphs:
            if p.parent:
                print(p.heading + f' (id:{p.id}) [parent id: {p.parent.id}]', '\n', p.paragraph, '\n')
            else:
                print(p.heading + f' (id:{p.id}) [root]', '\n', p.paragraph, '\n')

    def extract_font_block_heights (self) -> list[float|int]:
        
        heights = set()

        with pdfplumber.open(self.file) as pdf:
            for page in pdf.pages: 
                for block in page.extract_words(): 
                    heights.add(block["height"])

        return heights

    def extract_paragraphs (self, doctype: str='pdf') -> list[Paragraph]:

        '''
        Returns a list of Paragraph objects in procedual order.
        '''

        paragraphs = []

        if doctype.lower() == 'pdf':

            with pdfplumber.open(self.file) as pdf: 

                previous_font_size = 0 
                paragraph = "" 
                heading = None
                for page in pdf.pages: 
                    for block in page.extract_words():
                        text = block["text"] 
                        font_size = block["height"] 
                        if font_size > previous_font_size: 
                            if not heading:
                                heading = text
                                continue
                            if paragraph: 
                                # format paragraph
                                fp = paragraph.strip()
                                # determine depth from font size
                                depth = self.height_map[font_size]
                                # append Paragraph object to paragraphs
                                paragraphs.append(Paragraph(depth, heading, fp)) 
                                # reset loop parameters
                                paragraph = ""
                                heading = text # update heading with new found heading
                        else: 
                            paragraph += " " + text 
                        previous_font_size = font_size
        
        return paragraphs

    def extract_headings_and_paragraphs (self) -> tuple[list[str]]:

        '''
        Returns a tuple (headings, paragraphs) where each element is a list
        of headings and paragraphs, respectively.
        '''

        with pdfplumber.open(self.file) as pdf: 

            headings = [] 
            paragraphs = []
            previous_font_size = 0 
            paragraph = "" 

            for page in pdf.pages: 
                for block in page.extract_words():
                    text = block["text"] 
                    font_size = block["height"] 
                    if font_size > previous_font_size: 
                        if paragraph: 
                            paragraphs.append(paragraph.strip()) 
                            headings.append(text) 
                            paragraph = ""
                    else: 
                        paragraph += " " + text 
                    previous_font_size = font_size
        
        return headings, paragraphs

class DocReader:

    '''
    The DocReader uses Doc to analyze and abstract the underlying document.
    '''

    def __init__(self, 
                 file: str|PosixPath,
                 blowtorch_client: "client|None"=None) -> None:
        
        # initialize client
        if not blowtorch_client:
            blowtorch_client = client(
                "Llama-3.2-3B-Instruct.Q3_K_M.gguf",
                "MaziyarPanahi/Llama-3.2-3B-Instruct-GGUF",
                chat_format='llama-3',
                device='cpu')
        self.client = blowtorch_client