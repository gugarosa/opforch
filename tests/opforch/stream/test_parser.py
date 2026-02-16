import torch

from opforch.stream import loader, parser


def test_parse_loader():
    X, Y = parser.parse_loader(torch.tensor([]))

    assert X is None
    assert Y is None

    try:
        data = torch.ones((4, 4), dtype=torch.float64)
        X, Y = parser.parse_loader(data)
    except:
        try:
            data = torch.ones((4, 4), dtype=torch.float64)
            data[3, 1] = 3
            X, Y = parser.parse_loader(data)
        except:
            csv = loader.load_csv("data/boat.csv")

            X, Y = parser.parse_loader(csv)

            assert X.shape == (100, 2)
            assert Y.shape == (100,)
