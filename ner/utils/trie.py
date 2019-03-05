import collections
class TrieNode:
    # Initialize your data structure here.
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_word = False

class Trie:
    """
    In fact, this Trie is a letter three.
    root is a fake node, its function is only the begin of a word, same as <bow>

    the the first layer is all the word's possible first letter, for example, '中国'
        its first letter is '中'
    the second the layer is all the word's possible second letter.
    and so on
    """
    def __init__(self, use_single):
        self.root = TrieNode()
        if use_single:
            self.min_len = 0
        else:
            self.min_len = 1

    def insert(self, word):
        
        current = self.root
        # Traversing all the letter in the chinese word, util the last letter
        for letter in word:
            current = current.children[letter]
        current.is_word = True

    def search(self, word):
        current = self.root
        for letter in word:
            current = current.children.get(letter)

            if current is None:
                return False
        return current.is_word

    def startsWith(self, prefix):
        current = self.root
        for letter in prefix:
            current = current.children.get(letter)
            if current is None:
                return False
        return True


    def enumerateMatch(self, word, space="_", backward=False):
        matched = []
        ## while len(word) > 1 does not keep character itself, while word keed character itself
        while len(word) > self.min_len:
            if self.search(word):
                matched.append(space.join(word[:]))

            # del the last letter one by one
            del word[-1]
        # the matched is all the possible sub-sequence word in the give sequence word
        return matched

