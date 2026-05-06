"""
Sample data generation script for testing the recommendation system.
Creates dummy product data with images and search queries.
"""
import os
import csv
import random
from pathlib import Path
from PIL import Image, ImageDraw
import argparse


def generate_dummy_images(output_dir: str, num_images: int = 100):
    """
    Generate dummy product images for testing.
    Creates simple colored rectangles with text labels.
    
    Args:
        output_dir: Directory to save images
        num_images: Number of images to generate
    """
    os.makedirs(output_dir, exist_ok=True)
    
    categories = ['shirt', 'pants', 'shoes', 'hat', 'bag', 'jacket', 'dress']
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'white']
    
    image_paths = []
    
    for i in range(num_images):
        # Create a simple image
        img = Image.new('RGB', (224, 224), color=(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)))
        draw = ImageDraw.Draw(img)
        
        # Add text
        category = random.choice(categories)
        color = random.choice(colors)
        text = f"{color} {category}"
        draw.text((50, 100), text, fill=(255, 255, 255))
        
        # Save image
        image_path = os.path.join(output_dir, f"product_{i:04d}.jpg")
        img.save(image_path)
        image_paths.append(image_path)
    
    return image_paths


def generate_sample_csv(output_path: str, num_samples: int = 100):
    """
    Generate a sample CSV file with product data and queries.
    
    Args:
        output_path: Path to save CSV file
        num_samples: Number of samples to generate
    """
    # Generate dummy images
    image_dir = os.path.dirname(output_path) or 'data'
    os.makedirs(image_dir, exist_ok=True)
    
    image_paths = generate_dummy_images(os.path.join(image_dir, 'images'), num_samples)
    
    # Generate sample data
    categories = ['shirt', 'pants', 'shoes', 'hat', 'bag', 'jacket', 'dress']
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'white']
    materials = ['cotton', 'polyester', 'wool', 'silk', 'leather', 'denim']
    sizes = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
    
    queries = [
        "찾고 있는 바지",
        "넓은 소매의 셔츠",
        "편한 신발",
        "따뜻한 모자",
        "가죽 가방",
        "겨울 자켓",
        "예쁜 드레스",
        "파란 옷",
        "캐주얼 패션",
        "정장 추천",
    ]
    
    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['content_text', 'image_path', 'query'])
        
        for i in range(num_samples):
            category = random.choice(categories)
            color = random.choice(colors)
            material = random.choice(materials)
            size = random.choice(sizes)
            
            content_text = f"고급 {color} {category}. 재질: {material}, 사이즈: {size}. 최고급 품질의 제품입니다."
            image_path = image_paths[i]
            query = random.choice(queries)
            
            writer.writerow([content_text, image_path, query])
    
    print(f"Generated sample data: {output_path}")
    print(f"Generated {num_samples} images in {os.path.join(image_dir, 'images')}")


def main():
    parser = argparse.ArgumentParser(description='Generate sample data for recommendation system')
    parser.add_argument(
        '--output',
        type=str,
        default='data/sample_data.csv',
        help='Output path for CSV file'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of samples to generate'
    )
    
    args = parser.parse_args()
    
    generate_sample_csv(args.output, args.num_samples)


if __name__ == '__main__':
    main()
