#written by Ankita patra 
#B-number -B01101280
#A recursive descent parser Little Language LL of programs over boolean expression
# with variables and conditional statements.
# The grammar is LL(1), suitable for predictive parsing.


#Import required lib and pkg 
import re 
import sys
import json
from typing import List, Dict, Optional

# Token class represents individual tokens with type, value, and position in the input.
class Token:
    """ A simple Token structure. """
    def __init__(self, type: str, val: str, pos: int):
        self.type = type
        self.val = val
        self.pos = pos

    def __str__(self):
        return f'{self.type}({self.val}) at {self.pos}'

# LexerError is raised when an unexpected character is encountered during tokenization.
class LexerError(Exception):
    """ Lexer error exception. """
    def __init__(self, pos: int, char: str):
        super().__init__(f"Unexpected character at position {pos}: '{char}'")
        self.pos = pos
        self.char = char

# Lexer class is responsible for converting input strings into tokens using regex rules.
class Lexer:
    """ A regex-based lexer/tokenizer. """
    def __init__(self, rules: List[tuple], skip_whitespace: bool = True):
        self.regex_parts = []
        self.group_type = {}
        # Prepare the regex rules for tokenization
        for idx, (regex, type) in enumerate(rules, start=1):
            if type is not None:
                groupname = f'GROUP{idx}'
                self.regex_parts.append(f'(?P<{groupname}>{regex})')
                self.group_type[groupname] = type
        self.regex = re.compile('|'.join(self.regex_parts))
        self.skip_whitespace = skip_whitespace
        self.re_ws_skip = re.compile(r'\s+')
        self.re_comment_skip = re.compile(r'#.*')

    # Input method to provide a string to be tokenized.
    def input(self, buf: str):
        self.buf = buf
        self.pos = 0

    # Returns the next token from the input string or None if no more tokens.
    def token(self) -> Optional[Token]:
        if self.pos >= len(self.buf):
            return None

        # Skip whitespace and comments
        while self.pos < len(self.buf):
            m = self.re_ws_skip.match(self.buf, self.pos)
            if m:
                self.pos = m.end()
                continue

            m = self.re_comment_skip.match(self.buf, self.pos)
            if m:
                self.pos = m.end()
                continue

            break
        if self.pos >= len(self.buf):
            return None

        # Match tokens using the compiled regex.
        m = self.regex.match(self.buf, self.pos)
        if m:
            groupname = m.lastgroup
            tok_type = self.group_type[groupname]
            tok = Token(tok_type, m.group(groupname), self.pos)
            self.pos = m.end()
            return tok

        # Raise LexerError if no valid token is found at current position.
        raise LexerError(self.pos, self.buf[self.pos])

    # Generator that yields all tokens from the input.
    def tokens(self):
        while True:
            tok = self.token()
            if tok is None:
                return
            yield tok

# Parser class processes a list of tokens and builds an abstract syntax tree (AST).
class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.current = 0

    # Peek at the current token without advancing the position.
    def peek(self) -> Optional[Token]:
        if self.current < len(self.tokens):
            return self.tokens[self.current]
        return None

    # Consume the next token if it matches the expected type, else raise SyntaxError.
    def consume(self, expected_type: str) -> Token:
        token = self.peek()
        if token and token.type == expected_type:
            self.current += 1
            return token
        else:
            raise SyntaxError(f"Expected {expected_type} but got {token.type if token else 'end of input'}")

    # Parse the entire program, returning a list of parsed statements.
    def parse_program(self) -> List[Dict]:
        program = []
        while self.current < len(self.tokens):
            if self.peek().type == 'fn':
                program.append(self.parse_definition())
            else:
                program.append(self.parse_expression())
        return program

    # Parse function definitions (starting with 'fn').
    def parse_definition(self) -> Dict:
        self.consume('fn')
        name = self.consume('ID').val
        formals = []
        if self.peek().type == 'LPAREN':
            self.consume('LPAREN')
            if self.peek().type == 'ID':
                while True:
                    formals.append(self.consume('ID').val)
                    if self.peek().type == 'COMMA':
                        self.consume('COMMA')
                    else:
                        break
            self.consume('RPAREN')
        body = self.parse_expression()
        return {"tag": "def", "name": name, "formals": formals, "body": body}

    # Parse an expression, which can be a function application or a logical expression.
    def parse_expression(self) -> Dict:
        if self.peek().type == 'ID' and self.current + 1 < len(self.tokens) and self.tokens[self.current + 1].type == 'LPAREN':
            return self.parse_function_application()
        return self.parse_or_expression()

    # Parse function calls.
    def parse_function_application(self) -> Dict:
        name = self.consume('ID').val
        self.consume('LPAREN')
        args = []
        if self.peek().type != 'RPAREN':
            while True:
                args.append(self.parse_expression())
                if self.peek().type == 'COMMA':
                    self.consume('COMMA')
                else:
                    break
        self.consume('RPAREN')
        return {"tag": "app", "name": name, "args": args}

    # Parse OR expressions (using `|`).
    def parse_or_expression(self) -> Dict:
        left = self.parse_and_expression()
        while self.peek() and self.peek().type == 'OR':
            operator = self.consume('OR').val
            right = self.parse_and_expression()
            left = {'rand1': left, 'rand2': right, 'tag': operator}
        return left

    # Parse AND expressions (using `&`).
    def parse_and_expression(self) -> Dict:
        left = self.parse_prefix_expression()
        while self.peek() and self.peek().type == 'AND':
            operator = self.consume('AND').val
            right = self.parse_prefix_expression()
            left = {'rand1': left, 'rand2': right, 'tag': operator}
        return left

    # Parse prefix expressions (such as NOT).
    def parse_prefix_expression(self) -> Dict:
        if self.peek() and self.peek().type == 'NOT':
            self.consume('NOT')
            operand = self.parse_primary_expression()
            return {'tag': 'NOT', 'rand1': operand}
        return self.parse_primary_expression()

    # Parse primary expressions (identifiers, boolean literals, or parenthesized expressions).
    def parse_primary_expression(self) -> Dict:
        if self.peek() and self.peek().type == 'ID':
            return {'name': self.consume('ID').val, 'tag': 'id'}
        if self.peek() and self.peek().type == 'BOOLEAN':
            return {'tag': 'bool', 'value': self.consume('BOOLEAN').val}
        if self.peek() and self.peek().type == 'LPAREN':
            self.consume('LPAREN')
            expr = self.parse_expression()
            self.consume('RPAREN')
            return expr
        raise SyntaxError(f"Unexpected token {self.peek().type} ('{self.peek().val}') at position {self.peek().pos}")

# Main function to handle input and output.
def main():
    input_text = sys.stdin.read().strip()
    # Define the rules for tokenization.
    rules = [
        (r'\btrue\b', 'BOOLEAN'),
        (r'\bfalse\b', 'BOOLEAN'),
        (r'[a-zA-Z_][a-zA-Z0-9_-]*', 'ID'),
        (r'\&', 'AND'),
        (r'\|', 'OR'),
        (r'~', 'NOT'),
        (r'\(', 'LPAREN'),
        (r'\)', 'RPAREN'),
        (r',', 'COMMA'),
        (r'\s+', None), # Ignore whitespace
        (r'#.*', None), # Ignore comments
    ]
   
    # Initialize the lexer and process the input.
    lexer = Lexer(rules, skip_whitespace=True)
    lexer.input(input_text)
    tokens = list(lexer.tokens())

    # Initialize the parser and parse the tokenized input.
    parser = Parser(tokens)
    result = parser.parse_program()

    # Pretty print the result with indentation as JSON.
    print(json.dumps(result, indent=4))

# Entry point to run the main function.
if __name__ == "__main__":
    main()