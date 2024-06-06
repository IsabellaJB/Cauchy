class Padre():
  
    caballo="negro"
    ojos="azules"

    def conducir_coche(self):
        print ("La persona sabe conducir coches")

class Hijo(Padre):
  
    def conducir_moto(self):
        print ("La persona sabe conducir moto")

persona=Hijo()

print(persona.caballo)

print(persona.ojos)

persona.conducir_coche()

persona.conducir_moto() 