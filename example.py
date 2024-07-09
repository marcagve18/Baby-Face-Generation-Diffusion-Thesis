from baby_face_generation_package import BabyGenerator 
seed = 16355036249119675404

generator = BabyGenerator()

# Seed is specified for reproducibility
generation = generator.generate("Frontal portrait of a white smiling baby", seed=seed)
# Saving first generated image
generation.image.save("intial.png")

# Generating edited face
edited = generator.edit(generation, "Frontal portrait of a sad white baby")
# Saving edited face
edited.image.save("edited.png")
