(define (domain HOOKS)
    (:requirements :strips :typing)
    (:types hook)
    (:predicates
        (hooked ?y - hook)
        (holdingLoop)
		(notHoldingLoop)
    )

    (:action request-loop
        :parameters ()
        :precondition (notHoldingLoop)
        :effect (and (holdingLoop) (not (notHoldingLoop)))
    )

    (:action place-loop
        :parameters (?x - hook)
        :precondition (holdingLoop)
        :effect (and (notHoldingLoop) (not (holdingLoop)) (hooked ?x))
    )
)
