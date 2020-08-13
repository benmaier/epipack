import unittest

import numpy as np
import sympy

from epipack.process_conversions import (
            processes_to_rates,
            transition_processes_to_rates,
            fission_processes_to_rates,
            fusion_processes_to_rates,
            transmission_processes_to_rates,
            processes_to_events,
            transition_processes_to_events,
            fission_processes_to_events,
            fusion_processes_to_events,
            transmission_processes_to_events,
        )

class ProcessConversionTest(unittest.TestCase):

    def test_exceptions(self):

        self.assertRaises(TypeError,processes_to_rates,
                    [("S","S","S")], ["S"]
                )
        self.assertRaises(TypeError,processes_to_rates,
                    [("S","S","S","S")], ["S"]
                )
        self.assertRaises(TypeError,processes_to_rates,
                    [("S","S","S","S","S")], ["S"]
                )
        self.assertRaises(TypeError,processes_to_rates,
                    [("S","S","S","S","S","S")], ["S"]
                )
        self.assertRaises(TypeError,processes_to_events,
                    [("S","S","S")], ["S"]
                )
        self.assertRaises(TypeError,processes_to_events,
                    [("S","S","S","S")], ["S"]
                )
        self.assertRaises(TypeError,processes_to_events,
                    [("S","S","S","S","S")], ["S"]
                )
        self.assertRaises(TypeError,processes_to_events,
                    [("S","S","S","S","S","S")], ["S"]
                )

        S, I = sympy.symbols("S I")

        processes_to_rates(
                    [(S,S,I)], [S], ignore_rate_position_checks = True 
                )
        self.assertRaises(TypeError,processes_to_rates,
                    [(S,S,S,S)], [S], ignore_rate_position_checks = True
                )
        self.assertRaises(ValueError,processes_to_rates,
                    [(S,S,S,S,S)], [S], ignore_rate_position_checks = True
                )
        self.assertRaises(ValueError,processes_to_rates,
                    [(S,I,S,I,S)], [S], ignore_rate_position_checks = True
                )
        processes_to_events(
                    [(S,S,S,S,I)], [S], ignore_rate_position_checks = True
                )
        processes_to_events(
                    [(S,S,I)], [S], ignore_rate_position_checks = True 
                )
        self.assertRaises(TypeError,processes_to_events,
                    [(S,S,S,S)], [S], ignore_rate_position_checks = True
                )
        self.assertRaises(ValueError,processes_to_events,
                    [(S,S,S,S,S)], [S], ignore_rate_position_checks = True
                )
        self.assertRaises(ValueError,processes_to_events,
                    [(S,I,S,I,S)], [S], ignore_rate_position_checks = True
                )
        processes_to_events(
                    [(S,S,S,S,I)], [S], ignore_rate_position_checks = True
                )

    def test_transitions(self):

        self.assertRaises(ValueError,transition_processes_to_rates,
                    [("S",1,"S")]
                )
        self.assertRaises(ValueError,transition_processes_to_rates,
                    [(None,1,None)]
                )
        linear_rates = transition_processes_to_rates([
                ("S", 1, "I"),
                ("S", 1, None),
                (None, 1, "I"),
            ])
        expected = [
                    ("S", "S", -1),
                    ("S", "I", +1),
                    ("S", "S", -1),
                    (None, "I", +1),
                ]

        assert(all(a==b for a, b in zip(linear_rates, expected)))

        self.assertRaises(ValueError,transition_processes_to_events,
                    [("S",1,"S")]
                )
        self.assertRaises(ValueError,transition_processes_to_events,
                    [(None,1,None)]
                )
        linear_events = transition_processes_to_events([
                ("S", 2, "I"),
                ("S", 2, None),
                (None, 2, "I"),
            ])
        expected = [
                    (("S",), 2, [("S", -1), ("I", +1)]),
                    (("S",), 2, [("S", -1), ]),
                    ((None,), 2,[( "I", +1)]),
                ]

        assert(str(linear_events) == str(expected))


    def test_fission(self):

        rates = fission_processes_to_rates([
                ( "A", 3.14, "B", "C"),
            ])
        expected = [
                ( "A", "A", -3.14 ),
                ( "A", "B", +3.14 ),
                ( "A", "C", +3.14 ),
                ]
        assert(all(a==b for a, b in zip(rates, expected)))

        events = fission_processes_to_events([
                ( "A", 3.14, "B", "C"),
            ])
        expected = [
                ( ("A",), 3.14, [("A", -1), ("B", +1), ("C", +1)]),
                ]
        assert(str(events) == str(expected))

    def test_fusion(self):

        rates = fusion_processes_to_rates([
                ( "A", "B", 3.14, "C"),
            ])
        expected = [
                ( "A", "B", "C", +3.14 ),
                ( "A", "B", "A", -3.14 ),
                ( "A", "B", "B", -3.14 ),
                ]
        assert(all(a==b for a, b in zip(rates, expected)))

        events = fusion_processes_to_events([
                ( "A", "B", 3.14, "C"),
            ])
        expected = [
                ( ("A", "B"), 3.14, [("C", +1), ("A", -1), ("B", -1)]),
                ]
        assert(str(events)==str(expected))

    def test_transmission(self):
        rates = transmission_processes_to_rates([
                ( "S", "I", 0.5, "I", "I"),    
                ( "S", "I", 0.5, "B", "C"),    
            ])
        expected = [
                ("I","S","S", -0.5),
                ("I","S","I", +0.5),
                ("S","I","I", -0.5),
                ("S","I","C", +0.5),
                ("S","I","S", -0.5),
                ("S","I","B", +0.5),
            ]
        assert(all(a==b for a, b in zip(rates, expected)))

        events = transmission_processes_to_events([
                ( "S", "I", 0.5, "I", "I"),    
                ( "S", "I", 0.5, "B", "C"),    
            ])
        expected = [
                (("I","S"),0.5,[("S", -1),("I",+1)]),
                (("S","I"),0.5,[("I", -1),("C",+1),("S",-1),("B",+1)]),
            ]

        assert(str(events)==str(expected))

    def test_all(self):
        
        rates = [
                ("S", 1, "I"),
                ("S", 1, None),
                (None, 1, "I"),
            ]
        expected = [
                    ("S", "S", -1),
                    ("S", "I", +1),
                    ("S", "S", -1),
                    (None, "I", +1),
                ]
        rates += [
                ( "A", 3.14, "B", "C"),
            ]
        expected += [
                ( "A", "A", -3.14 ),
                ( "A", "B", +3.14 ),
                ( "A", "C", +3.14 ),
            ]
        rates += [
                ( "A", "B", 3.14, "C"),
            ]
        expected += [
                ( "A", "B", "C", +3.14 ),
                ( "A", "B", "A", -3.14 ),
                ( "A", "B", "B", -3.14 ),
            ]

        rates += [
                ( "S", "I", 0.5, "I", "I"),    
                ( "S", "I", 0.5, "B", "C"),    
            ]
        expected += [
                ("I","S","S", -0.5),
                ("I","S","I", +0.5),
                ("S","I","I", -0.5),
                ("S","I","C", +0.5),
                ("S","I","S", -0.5),
                ("S","I","B", +0.5),
            ]

        qrates, lrates = processes_to_rates(rates,["S","I","A","B","C"])
        rates = set(lrates + qrates)
        expected = set(expected)

        assert(rates == expected)

        events = [
                ("S", 2, "I"),
                ("S", 2, None),
                (None, 2, "I"),
            ]
        lexpected = [
                    (("S",), 2, [("S", -1), ("I", +1)]),
                    (("S",), 2, [("S", -1), ]),
                    ((None,), 2,[( "I", +1)]),
                ]
        events += [
                ( "A", 3.14, "B", "C"),
            ]
        lexpected += [
                ( ("A",), 3.14, [("A", -1), ("B", +1), ("C", +1)]),
            ]
        events += [
                ( "A", "B", 3.14, "C"),
            ]
        qexpected = [
                ( ("A", "B"), 3.14, [("C", +1), ("A", -1), ("B", -1)]),
            ]

        events += [
                ( "S", "I", 0.5, "I", "I"),    
                ( "S", "I", 0.5, "B", "C"),    
            ]
        qexpected += [
                (("I","S"),0.5,[("S", -1),("I",+1)]),
                (("S","I"),0.5,[("I", -1),("C",+1),("S",-1),("B",+1)]),
            ]

        qevents, levents = processes_to_events(events,["S","I","A","B","C"])
        events = levents + qevents
        expected = lexpected + qexpected

        assert(str(events) == str(expected))




if __name__ == "__main__":

    T = ProcessConversionTest()
    T.test_exceptions()
    T.test_transitions()
    T.test_fission()
    T.test_fusion()
    T.test_transmission()
    T.test_all()
