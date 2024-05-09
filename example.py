from baby_face_generation_package import BabyGenerator 

seed = 16355036249119675404

generator = BabyGenerator()

generation = generator.generate("Frontal portrait of a white smiling baby", seed=seed)

edited = generator.edit(generation, "Frontal portrait of a sad white baby")