from genlm.control.simulation_utils import even_bit_set_number


def test_even_bits():
    assert even_bit_set_number(10) == 10
    assert even_bit_set_number(20) == 30
    assert even_bit_set_number(30) == 30
