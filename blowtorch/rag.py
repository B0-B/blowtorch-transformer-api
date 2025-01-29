from pathlib import PosixPath, Path
from blowtorch import client
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

        if 'pdf' in doctype.lower():

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

    backup_model = "MaziyarPanahi/Llama-3.2-3B-Instruct-GGUF"
    backup_model_file = "Llama-3.2-3B-Instruct.Q3_K_M.gguf"

    def __init__(self, 
                 file: str|PosixPath,
                 abstraction_map: list[int],
                 blowtorch_client: "client|None"=None) -> None:
        
        # initialize client if not provided
        if not blowtorch_client:
            print(f'[{self.__class__.__name__}] ⚠️ No client provided, will initilize one from {DocReader.backup_model}.')
            blowtorch_client = client(
                DocReader.backup_model_file,
                DocReader.backup_model,
                chat_format='llama-3',
                device='cpu')
        self.client = blowtorch_client

        # client inference kwargs
        self.output_length = 512
        self.input_length = 2000
        self.temperature = 1.05

        # The Doc instance holds all paragraphs, relational tree path 
        # and other document information
        self.document = Doc(file)

        # Prepared prompting template.
        self.system_prompt = 'You evaluate, comprehend and summarize paragraphs of a document or paper. The paragraphs can be text, or data.'
        self.prompt_template = 'Abstract the following text in {} sentences but include everything:\n\n{}' 

        # abstract all paragraphs using the abstraction map
        self.abstraction_map = abstraction_map
        self.abstract_paragraphs(abstraction_map)

        # init context
        self.context_id = 0
        self.client.newConversation(self.context_id, 'user', scenario='You are a helpful analyst and assistant which studies texts from papers and documents to explain it to the user for easier comprehension.')

    def abstract_paragraphs (self, abstraction_map: list[int], **pipe_twargs) -> None:

        '''
        Abstracts all paragraphs to the size defined in abstraction map.
        The abstraction map defines in how many sentences the paragraph (full text)
        should be summarized (or abstracted). 

        [Parameter]
        
        abstraction_map :           List of integer numbers e.g. [8,4,2] summarizes
                                    each paragraph in 8, 4 and 2 sentences.
        '''

        # clarify twargs by merging with config (if enabled)
        if self.client.config:

            # override pipe twargs with left twargs in config
            pipe_twargs.update(self.client.config)

        for sentence_count in abstraction_map:

            # vLLM allows to parallelize requests by vectorizing
            if self.client.llm_base_module == 'vllm':

                # vectorize all paragraphs to one input vector
                _inputs = []
                for paragraph in self.document.paragraphs:
                    prompt_text = self.prompt_template.format(sentence_count, paragraph.paragraph)
                    formatted_prompt = self.client.__format_prompt__(prompt_text, 'user', system_prompt=self.system_prompt)
                    _inputs.append(formatted_prompt)

                # batched forward propagation
                _outputs = self.client.batch_inference(*_inputs, **pipe_twargs)

                # sort in answers into paragraph objects 
                for i in range(len(_outputs)):
                    
                    answer = _outputs[i]
                    paragraph = self.document.paragraphs[i]

                    paragraph.abstraction.append(answer)
            
            # Otherwise compute all paragraphs sequentially.
            else:

                # Iterate sequentially over the documents paragraphs
                for paragraph in self.document.paragraphs:

                    prompt_text = self.prompt_template.format(sentence_count, paragraph.paragraph)
                    formatted_prompt = self.client.__format_prompt__(prompt_text, 'user', system_prompt=self.system_prompt)

                    answer = self.client.inference(formatted_prompt, **pipe_twargs)
                    paragraph.abstraction.append(answer)
    
    def summary (self) -> str:

        depth = 0
        merged_abstractions = '\n'.join([paragraph.abstraction[depth] for paragraph in self.document.paragraphs])
        prompt = f'The following are sequentially merged summaries of all paragraphs of a paper, please comprehend and summarize it fluently:\n\n{merged_abstractions}'

        return self.client.contextInference(prompt, sessionId=self.context_id, auto_trim=True, max_new_tokens=self.output_length, temperature=self.temperature)