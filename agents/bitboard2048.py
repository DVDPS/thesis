import numpy as np
import random

def unpack_row(row):
    """Extract four 4-bit values from a 16-bit row integer."""
    return [(row >> (4 * i)) & 0xF for i in range(4)]

def pack_row(tiles):
    """Pack a list of 4 4-bit tile values into a 16-bit integer."""
    result = 0
    for i, tile in enumerate(tiles):
        result |= (tile & 0xF) << (4 * i)
    return result

def move_row_left(row):
    """
    Given a 16-bit row, simulate a left move (merging adjacent identical tiles).
    Tiles are stored as exponents (0 means empty, 1 means 2, 2 means 4, etc.).
    Returns (new_row, score), where score is the sum of merged tile values.
    """
    tiles = unpack_row(row)
    new_tiles = [tile for tile in tiles if tile != 0]
    score = 0
    i = 0
    while i < len(new_tiles) - 1:
        if new_tiles[i] != 0 and new_tiles[i] == new_tiles[i+1]:
            new_tiles[i] += 1  # Merge by increasing the exponent.
            score += (1 << new_tiles[i])  # Actual tile value: 2^(exponent)
            del new_tiles[i+1]
            new_tiles.append(0)
        i += 1
    new_tiles = [tile for tile in new_tiles if tile != 0] + [0] * (4 - len(new_tiles))
    new_row = pack_row(new_tiles)
    return new_row, score

# Precompute lookup table for all 16-bit rows.
ROW_LOOKUP = {}
for row in range(65536):
    new_row, score = move_row_left(row)
    ROW_LOOKUP[row] = (new_row, score)

class Bitboard2048:
    """
    Bitboard representation for a 2048 board.
    The board is stored as a 64-bit integer; each row occupies 16 bits.
    """
    def __init__(self, board=None):
        self.board = np.uint64(board) if board is not None else np.uint64(0)

    def get_row(self, r):
        """Extract row r (0-indexed, where row 0 is the least-significant 16 bits)."""
        return (self.board >> (16 * r)) & 0xFFFF

    def set_row(self, r, row_value):
        """Set row r to a given 16-bit integer value."""
        mask = np.uint64(0xFFFF) << (16 * r)
        self.board = (self.board & ~mask) | (np.uint64(row_value) << (16 * r))

    def move_left(self):
        """Apply a left move to each row using the lookup table."""
        new_board = np.uint64(0)
        total_score = 0
        for r in range(4):
            row = int(self.get_row(r))
            new_row, score = ROW_LOOKUP[row]
            total_score += score
            new_board |= (np.uint64(new_row) << (16 * r))
        return Bitboard2048(new_board), total_score

    def move_right(self):
        """Move right by reversing each row, moving left, then reversing back."""
        new_board = np.uint64(0)
        total_score = 0
        for r in range(4):
            row = int(self.get_row(r))
            tiles = unpack_row(row)
            tiles.reverse()
            rev_row = pack_row(tiles)
            new_rev, score = ROW_LOOKUP[rev_row]
            total_score += score
            new_tiles = unpack_row(new_rev)
            new_tiles.reverse()
            final_row = pack_row(new_tiles)
            new_board |= (np.uint64(final_row) << (16 * r))
        return Bitboard2048(new_board), total_score

    def transpose(self):
        """Transpose the 4x4 board by converting to a 2D list and back."""
        mat = [[(int(self.board) >> (16 * r + 4 * c)) & 0xF for c in range(4)] for r in range(4)]
        transposed_mat = list(map(list, zip(*mat)))
        new_board = np.uint64(0)
        for r in range(4):
            row_val = pack_row(transposed_mat[r])
            new_board |= (np.uint64(row_val) << (16 * r))
        self.board = new_board

    def move_up(self):
        """Move up by transposing, moving left, then transposing back."""
        self.transpose()
        moved, score = self.move_left()
        moved.transpose()
        return moved, score

    def move_down(self):
        """Move down by transposing, moving right, then transposing back."""
        self.transpose()
        moved, score = self.move_right()
        moved.transpose()
        return moved, score

    def copy(self):
        return Bitboard2048(self.board)

    def to_numpy(self):
        """
        Convert the bitboard to a 4x4 numpy array of actual tile values.
        Tiles are reconstructed from the exponent: tile = 2^(exponent), with 0 representing empty.
        """
        arr = np.zeros((4, 4), dtype=np.int32)
        for r in range(4):
            row = unpack_row(int(self.get_row(r)))
            for c in range(4):
                exp = row[c]
                arr[r, c] = 0 if exp == 0 else (1 << exp)
        return arr

    def __str__(self):
        arr = self.to_numpy()
        return "\n".join(["|".join([f"{v:4}" for v in row]) for row in arr])

def add_random_tile(bitboard: Bitboard2048) -> Bitboard2048:
    """Add a random tile (2 or 4) to an empty cell"""
    # Get list of empty cells
    empty_cells = []
    state = bitboard.to_numpy()
    for i in range(4):
        for j in range(4):
            if state[i, j] == 0:
                empty_cells.append((i, j))
    
    if not empty_cells:
        return bitboard
    
    # Choose a random empty cell
    i, j = empty_cells[np.random.randint(len(empty_cells))]
    
    # Choose value (2 with 90% probability, 4 with 10% probability)
    value = 2 if np.random.random() < 0.9 else 4
    
    # Convert value to exponent (2 -> 1, 4 -> 2)
    exponent = int(value).bit_length() - 1
    
    # Add the tile to the bitboard
    new_board = np.uint64(bitboard.board)  # Ensure we're working with uint64
    row_mask = np.uint64(0xF) << np.uint64(j * 4)  # Mask for the 4 bits in the row
    new_board &= ~(row_mask << np.uint64(i * 16))  # Clear the bits for this position
    new_board |= (np.uint64(exponent & 0xF) << np.uint64(j * 4 + i * 16))  # Set the new value
    
    return Bitboard2048(new_board)

# Example usage:
if __name__ == "__main__":
    # Start with an empty bitboard.
    bb = Bitboard2048()
    bb = add_random_tile(bb)
    bb = add_random_tile(bb)
    print("Initial board:")
    print(bb)
    
    # Apply a left move.
    bb, score = bb.move_left()
    print("\nAfter move left, score gained:", score)
    print(bb)
    
    # For demonstration, you can also try other moves:
    bb = add_random_tile(bb)
    bb, score = bb.move_up()
    print("\nAfter move up, score gained:", score)
    print(bb) 