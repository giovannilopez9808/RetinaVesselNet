# Reconocimiento de patrones proyecto 02

## Retina VesselNet

### Organización de archivos

```bash
├── organization.txt
├── test
│  ├── high_contrast
│  │  └── data
│  │     ├── DRIVE_test_001.jpg
│  │     ├── ...
│  │     └── DRIVE_test_020.jpg
│  ├── label
│  │  └── data
│  │     ├── DRIVE_test_001_mask.jpg
│  │     ├── ...
│  │     └── DRIVE_test_020_mask.jpg
│  └── normal
│     └── data
│        ├── DRIVE_test_001.jpg
│        ├── ...
│        └── DRIVE_test_020.jpg
├── train
│  ├── high_contrast
│  │  └── data
│  │     ├── DRIVE_0001.jpg
│  │     ├── ...
│  │     └── DRIVE_1008.jpg
│  ├── label
│  │  └── data
│  │     ├── DRIVE_0001_mask.png
│  │     ├── ...
│  │     └── DRIVE_1008_mask.png
│  └── normal
│     └── data
│        ├── DRIVE_0001.jpg
│        ├── ...
│        └── DRIVE_1008.jpg
└── validate
   ├── high_contrast
   │  └── data
   │    ├── DRIVE_validate_001.jpg
   │    ├── ...
   │    └── DRIVE_validate_252.jpg
   ├── label
   │  └── data
   │     ├── DRIVE_validate_001_mask.png
   │     ├── ...
   │     └── DRIVE_validate_252_mask.png
   └── normal
      └── data
         ├── DRIVE_validate_001.jpg
         ├── ...
         └── DRIVE_validate_252.jpg

```

### Alto contraste

Se utilizo un script para generar las imágenes guardadas en la carpeta `high_contrast` a partir de las imágenes originales

### Modelos preentrenados

Los modelos se encuentran en la carpeta `results`
