"""
Project Aether - Data Loading Utilities
Handles I2P (unsafe) and MS-COCO (safe) datasets.
"""

import os
import json
import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# CLIP tokenizer for checking prompt length
try:
    from transformers import CLIPTokenizer
    _clip_tokenizer = None
    
    def get_clip_tokenizer():
        """Lazy load CLIP tokenizer."""
        global _clip_tokenizer
        if _clip_tokenizer is None:
            _clip_tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14",
                cache_dir="./data/cache"
            )
        return _clip_tokenizer
    
    def count_clip_tokens(prompt: str) -> int:
        """Count CLIP tokens in a prompt (max 77)."""
        try:
            tokenizer = get_clip_tokenizer()
            tokens = tokenizer(prompt, truncation=False, return_tensors="pt")
            return tokens.input_ids.shape[1]
        except Exception:
            # Fallback: estimate ~1.3 tokens per word
            return int(len(prompt.split()) * 1.3)
    
    def is_prompt_valid(prompt: str, max_tokens: int = 77) -> bool:
        """Check if prompt fits within CLIP token limit."""
        return count_clip_tokens(prompt) <= max_tokens
except ImportError:
    # Fallback if transformers not available
    def count_clip_tokens(prompt: str) -> int:
        """Estimate tokens: ~1.3 tokens per word."""
        return int(len(prompt.split()) * 1.3)
    
    def is_prompt_valid(prompt: str, max_tokens: int = 77) -> bool:
        """Check if prompt fits within CLIP token limit (estimated)."""
        return count_clip_tokens(prompt) <= max_tokens


@dataclass
class DataConfig:
    """Configuration for data loading."""
    i2p_categories: List[str] = None  # Filter I2P by categories
    num_safe_samples: int = 500
    num_unsafe_samples: int = 500
    random_seed: int = 42
    cache_dir: str = "./data/cache"
    
    def __post_init__(self):
        if self.i2p_categories is None:
            # Focus on high-harm concepts as per your project spec
            self.i2p_categories = ["violence", "sexual", "shocking"]


class I2PDataset:
    """
    Loader for the I2P (Inappropriate Image Prompts) benchmark.
    Source: https://huggingface.co/datasets/AIML-TUDA/i2p
    
    This dataset contains prompts that are likely to generate inappropriate content.
    Categories: hate, harassment, violence, self-harm, sexual, shocking, illegal activity
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self._dataset = None
        
    def load(self) -> None:
        """Load the I2P dataset from HuggingFace."""
        print("Loading I2P dataset from HuggingFace...")
        self._dataset = load_dataset(
            "AIML-TUDA/i2p",
            split="train",
            cache_dir=self.config.cache_dir
        )
        print(f"Loaded {len(self._dataset)} prompts from I2P")
        
    def get_prompts(
        self,
        categories: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        hard_only: bool = False,
        min_inappropriate_pct: float = 30.0,
    ) -> List[Dict]:
        """
        Get filtered prompts from I2P.
        
        Args:
            categories: Filter by categories (e.g., ["violence", "sexual"])
            max_samples: Maximum number of samples to return
            hard_only: Only return "hard" prompts (>50% inappropriate)
            min_inappropriate_pct: Minimum inappropriate percentage threshold
            
        Returns:
            List of prompt dictionaries with metadata
        """
        if self._dataset is None:
            self.load()
            
        categories = categories or self.config.i2p_categories
        max_samples = max_samples or self.config.num_unsafe_samples
        
        filtered = []
        skipped_long = 0
        for item in self._dataset:
            # Check category filter
            item_cats = item["categories"].lower()
            if categories:
                if not any(cat.lower() in item_cats for cat in categories):
                    continue
            
            # Check hard filter
            if hard_only and item["hard"] != 1:
                continue
                
            # Check inappropriate percentage
            if item["inappropriate_percentage"] < min_inappropriate_pct:
                continue
            
            # Check CLIP token limit (skip instead of truncate)
            if not is_prompt_valid(item["prompt"], max_tokens=77):
                skipped_long += 1
                continue
                
            filtered.append({
                "prompt": item["prompt"],
                "categories": item["categories"],
                "inappropriate_pct": item["inappropriate_percentage"],
                "nudity_pct": item["nudity_percentage"],
                "is_hard": item["hard"] == 1,
                "label": 1,  # Unsafe
            })
            
            # Stop when we have enough (with some buffer for filtering)
            if len(filtered) >= max_samples * 2:
                break
        
        # Shuffle and limit
        random.seed(self.config.random_seed)
        random.shuffle(filtered)
        
        result = filtered[:max_samples]
        if skipped_long > 0:
            print(f"Selected {len(result)} unsafe prompts from I2P (skipped {skipped_long} prompts exceeding 77 CLIP tokens)")
        else:
            print(f"Selected {len(result)} unsafe prompts from I2P")
        return result
    
    def get_category_distribution(self) -> Dict[str, int]:
        """Get distribution of categories in the dataset."""
        if self._dataset is None:
            self.load()
            
        dist = {}
        for item in self._dataset:
            for cat in item["categories"].split(", "):
                cat = cat.strip().lower()
                dist[cat] = dist.get(cat, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: -x[1]))


class COCOCaptions:
    """
    Loader for MS-COCO captions as safe prompts.
    Source: https://cocodataset.org/
    
    Uses the HuggingFace datasets library for easy access.
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self._dataset = None
        
    def load(self) -> None:
        """Load COCO captions from HuggingFace."""
        print("Loading MS-COCO captions from HuggingFace...")
        # Using the COCO captions dataset from HuggingFace
        self._dataset = load_dataset(
            "HuggingFaceM4/COCO",
            "2017_captions",
            split="train",
            cache_dir=self.config.cache_dir,
            trust_remote_code=True
        )
        print(f"Loaded {len(self._dataset)} captions from MS-COCO")
        
    def get_prompts(
        self,
        max_samples: Optional[int] = None,
        min_length: int = 20,
        max_length: int = 200,
    ) -> List[Dict]:
        """
        Get safe prompts from COCO captions.
        
        Args:
            max_samples: Maximum number of samples
            min_length: Minimum caption length
            max_length: Maximum caption length
            
        Returns:
            List of prompt dictionaries
        """
        if self._dataset is None:
            self.load()
            
        max_samples = max_samples or self.config.num_safe_samples
        
        filtered = []
        seen_prompts = set()
        
        for item in self._dataset:
            # COCO has multiple captions per image - get the first one
            if "sentences" in item:
                captions = item["sentences"]["raw"]
            elif "caption" in item:
                captions = [item["caption"]] if isinstance(item["caption"], str) else item["caption"]
            else:
                continue
                
            for caption in captions:
                if caption in seen_prompts:
                    continue
                    
                # Length filter
                if len(caption) < min_length or len(caption) > max_length:
                    continue
                
                # Check CLIP token limit (skip instead of truncate)
                if not is_prompt_valid(caption, max_tokens=77):
                    continue
                    
                seen_prompts.add(caption)
                filtered.append({
                    "prompt": caption,
                    "categories": "safe",
                    "label": 0,  # Safe
                })
                
                if len(filtered) >= max_samples * 2:  # Get extra for shuffling
                    break
            
            if len(filtered) >= max_samples * 2:
                break
        
        # Shuffle and limit
        random.seed(self.config.random_seed)
        random.shuffle(filtered)
        
        result = filtered[:max_samples]
        if len(result) < max_samples:
            print(f"Warning: Only found {len(result)} valid safe prompts (requested {max_samples})")
        print(f"Selected {len(result)} safe prompts from MS-COCO")
        return result


class AlternativeSafePrompts:
    """
    Fallback safe prompts if COCO loading is slow/problematic.
    These are curated, clearly safe prompts covering diverse categories.
    Expanded to 200+ base prompts for better training diversity.
    """
    
    SAFE_PROMPTS = [
        # Nature & Landscapes (40 prompts)
        "a beautiful sunset over the ocean with orange and pink clouds",
        "a serene mountain landscape with snow-capped peaks",
        "a peaceful lake surrounded by autumn trees",
        "a field of sunflowers under blue sky",
        "a rainbow appearing after a summer rain",
        "a waterfall in a tropical rainforest",
        "a hot spring in a forest",
        "a beautiful aurora borealis over snowy mountains",
        "a desert landscape with sand dunes at golden hour",
        "a misty forest at dawn with rays of sunlight",
        "rolling green hills in the countryside",
        "a tropical beach with crystal clear water",
        "a volcanic crater lake with turquoise water",
        "a lavender field in Provence during summer",
        "a redwood forest with towering ancient trees",
        "a glacier reflecting in a calm alpine lake",
        "autumn foliage in a New England forest",
        "a meadow of wildflowers in spring",
        "a rocky coastline with crashing waves",
        "a river winding through a canyon",
        "a bamboo forest in Japan",
        "northern lights over a frozen lake",
        "a savanna landscape with acacia trees",
        "cherry blossoms along a river in spring",
        "a cliff overlooking the ocean",
        "rolling fog over San Francisco bay",
        "a vineyard in Tuscany at sunset",
        "a frozen waterfall in winter",
        "sand dunes under a starry night sky",
        "a mountain reflected in a still pond",
        "a coral atoll from above",
        "storm clouds over prairie grasslands",
        "a hidden cove with turquoise water",
        "moss-covered stones in a Japanese garden",
        "a field of tulips in the Netherlands",
        "sunrise over rice terraces",
        "a Scottish highlands landscape",
        "palm trees on a Caribbean beach",
        "autumn leaves floating on a pond",
        "a peaceful zen garden with raked gravel",
        
        # Animals (40 prompts)
        "a fluffy cat sleeping peacefully on a cozy couch",
        "a golden retriever running on a sandy beach",
        "a cute puppy playing with a tennis ball",
        "a baby elephant playing in water",
        "a colorful parrot perched on a branch",
        "a fox in autumn leaves",
        "a sleepy owl on a tree branch",
        "a kitten playing with yarn",
        "a beautiful coral reef with tropical fish",
        "a hummingbird hovering near red flowers",
        "a colorful butterfly resting on a blooming flower",
        "a panda eating bamboo in a forest",
        "a lion family resting in the savanna",
        "a penguin colony on Antarctic ice",
        "a deer in a snowy forest",
        "a dolphin jumping out of ocean water",
        "a koala sleeping in a eucalyptus tree",
        "a red fox in a winter landscape",
        "a peacock displaying colorful feathers",
        "a sea turtle swimming in clear water",
        "a family of ducks swimming in a pond",
        "a horse running through a field",
        "a bunny in a flower garden",
        "a whale breaching the ocean surface",
        "a flamingo standing in shallow water",
        "a squirrel gathering acorns",
        "a hedgehog in autumn leaves",
        "a swan on a calm lake",
        "a giraffe eating from a tall tree",
        "a seal basking on a rock",
        "a monarch butterfly on a milkweed plant",
        "a robin perched on a branch in spring",
        "a border collie playing fetch",
        "a maine coon cat with fluffy fur",
        "a frog on a lily pad",
        "a ladybug on a green leaf",
        "a hamster eating seeds",
        "a tropical fish in an aquarium",
        "a bee collecting pollen from flowers",
        "a arctic fox in snowy landscape",
        
        # Food & Cooking (30 prompts)
        "a stack of pancakes with maple syrup and berries",
        "a delicious pizza fresh from the oven",
        "a cup of hot chocolate with marshmallows",
        "a bowl of fresh fruit salad",
        "a professional photograph of fresh fruits on a wooden table",
        "a baker decorating a birthday cake",
        "a chef preparing a gourmet meal in a kitchen",
        "sushi arranged beautifully on a wooden board",
        "a steaming bowl of ramen noodles",
        "fresh croissants on a bakery counter",
        "a colorful smoothie bowl with toppings",
        "grilled vegetables on a summer barbecue",
        "a chocolate lava cake with ice cream",
        "a rustic bread loaf fresh from the oven",
        "a farmers market fruit display",
        "a plate of spaghetti with tomato sauce",
        "a refreshing lemonade with mint leaves",
        "a charcuterie board with cheese and crackers",
        "cupcakes with colorful frosting",
        "a bowl of homemade soup with bread",
        "a cup of cappuccino with latte art",
        "macarons in pastel colors",
        "a thanksgiving turkey dinner",
        "fresh baked cookies on a cooling rack",
        "a fruit tart with glazed berries",
        "a picnic basket with sandwiches",
        "a waffle with fresh strawberries",
        "dim sum in bamboo steamers",
        "a colorful acai bowl",
        "homemade apple pie cooling on windowsill",
        
        # Architecture & Places (30 prompts)
        "a cozy coffee shop interior with warm lighting",
        "a modern city skyline at dusk with lights",
        "a lighthouse on a rocky coast at sunset",
        "a cozy cabin in a snowy forest",
        "a snow-covered village in winter",
        "a street cafe in Paris",
        "a wooden bridge over a stream",
        "a vintage bicycle parked near a flower shop",
        "the Eiffel Tower at night with lights",
        "a quaint English cottage with garden",
        "a traditional Japanese temple",
        "a Venetian canal with gondolas",
        "a colorful street in Santorini Greece",
        "a cozy library with floor to ceiling books",
        "a modern minimalist living room",
        "a Victorian house with wrap around porch",
        "a rustic barn in countryside",
        "a medieval castle on a hilltop",
        "a treehouse in a lush forest",
        "a beach house with ocean view",
        "a charming Amsterdam canal house",
        "a Moroccan courtyard with tiles",
        "a Swiss chalet in the Alps",
        "a rooftop garden in a city",
        "a covered bridge in autumn",
        "a windmill in Dutch countryside",
        "a stone cottage in Ireland",
        "a houseboat on a calm river",
        "a modern glass skyscraper",
        "a historic downtown street",
        
        # Activities & People (30 prompts)
        "children playing happily in a sunny park with green grass",
        "a family having a picnic in a meadow",
        "a musician playing guitar in a park",
        "a street artist painting a mural",
        "a child blowing soap bubbles",
        "a couple walking on a beach at sunset",
        "a farmer feeding chickens",
        "a yoga practitioner in a peaceful pose",
        "a child flying a colorful kite",
        "friends camping under the stars",
        "a potter making ceramics on a wheel",
        "a grandmother reading to grandchildren",
        "a cyclist riding through countryside",
        "children building a sandcastle",
        "a gardener tending to flowers",
        "a fisherman at sunset",
        "a family decorating a Christmas tree",
        "a painter at an easel outdoors",
        "children playing in autumn leaves",
        "a surfer riding a wave",
        "a rock climber on a cliff face",
        "a hiker at a mountain summit",
        "a skier on fresh powder snow",
        "a child learning to ride a bicycle",
        "friends having a picnic in a park",
        "a dancer practicing ballet",
        "a chef teaching a cooking class",
        "a florist arranging a bouquet",
        "children at a lemonade stand",
        "a family playing board games",
        
        # Objects & Still Life (30 prompts)
        "a bookshelf filled with colorful books",
        "a hot air balloon floating over green valleys",
        "a Christmas tree decorated with lights",
        "a beautiful garden with roses and tulips in bloom",
        "a vintage camera on a wooden desk",
        "antique pocket watch on velvet",
        "a crystal vase with fresh flowers",
        "a stack of old leather-bound books",
        "colorful balloons against blue sky",
        "a vintage typewriter on a desk",
        "a compass on an old map",
        "art supplies scattered on a table",
        "a collection of seashells",
        "a vintage record player",
        "a telescope pointed at the night sky",
        "a chess set mid-game",
        "candles and flowers on a table",
        "a handwritten letter with wax seal",
        "musical instruments in a studio",
        "a snow globe with winter scene",
        "a kite flying in blue sky",
        "a jar of fireflies at dusk",
        "a stack of wrapped presents",
        "a vintage clock on mantelpiece",
        "a basket of freshly picked apples",
        "a hammock between two trees",
        "a bicycle with basket of flowers",
        "a treehouse with rope ladder",
        "a cozy reading nook by window",
        "paper lanterns at a festival",
    ]
    
    # Style modifiers for augmentation
    STYLE_MODIFIERS = [
        "a photograph of", "a painting of", "an illustration of",
        "a digital art of", "a realistic image of", "a beautiful",
        "a professional photo of", "an artistic rendering of",
        "a high quality image of", "a stunning view of",
        "a cinematic shot of", "a detailed illustration of",
    ]
    
    # Quality modifiers for augmentation
    QUALITY_MODIFIERS = [
        "highly detailed", "award winning photography",
        "8k resolution", "masterpiece", "beautiful lighting",
        "professional quality", "vibrant colors",
    ]
    
    @classmethod
    def get_prompts(cls, num_samples: int = 200, seed: int = 42) -> List[Dict]:
        """
        Get safe prompts with optional augmentation.
        
        Args:
            num_samples: Number of prompts to return
            seed: Random seed for reproducibility
            
        Returns:
            List of prompt dictionaries
        """
        random.seed(seed)
        
        # Start with base prompts (filter by CLIP tokens)
        augmented = []
        for prompt in cls.SAFE_PROMPTS:
            if is_prompt_valid(prompt, max_tokens=77):
                augmented.append({
                    "prompt": prompt, 
                    "categories": "safe", 
                    "label": 0,
                    "augmented": False,
                })
        
        # Add augmented versions if we need more samples
        if num_samples > len(augmented):
            for prompt in cls.SAFE_PROMPTS:
                # Style prefix augmentation
                for mod in random.sample(cls.STYLE_MODIFIERS, min(3, len(cls.STYLE_MODIFIERS))):
                    augmented_prompt = f"{mod} {prompt}"
                    # Check CLIP token limit for augmented prompts
                    if is_prompt_valid(augmented_prompt, max_tokens=77):
                        augmented.append({
                            "prompt": augmented_prompt,
                            "categories": "safe",
                        "label": 0,
                        "augmented": True,
                    })
                
                # Quality suffix augmentation
                quality = random.choice(cls.QUALITY_MODIFIERS)
                quality_prompt = f"{prompt}, {quality}"
                # Check CLIP token limit for quality-augmented prompts
                if is_prompt_valid(quality_prompt, max_tokens=77):
                    augmented.append({
                        "prompt": quality_prompt,
                        "categories": "safe", 
                        "label": 0,
                        "augmented": True,
                    })
        
        random.shuffle(augmented)
        result = augmented[:num_samples]
        if len(result) < num_samples:
            print(f"Warning: Only found {len(result)} valid safe prompts (requested {num_samples}) after CLIP token filtering")
        print(f"Selected {len(result)} safe prompts ({sum(1 for p in result if not p.get('augmented', False))} base, {sum(1 for p in result if p.get('augmented', False))} augmented)")
        return result
    
    @classmethod
    def get_base_count(cls) -> int:
        """Return the number of base (non-augmented) prompts."""
        return len(cls.SAFE_PROMPTS)


def load_prompt_dataset(
    config: Optional[DataConfig] = None,
    use_coco: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load both safe and unsafe prompt datasets.
    
    Args:
        config: Data configuration
        use_coco: Whether to use COCO (if False, uses curated safe prompts)
        
    Returns:
        Tuple of (safe_prompts, unsafe_prompts)
    """
    config = config or DataConfig()
    
    # Load unsafe prompts from I2P
    i2p = I2PDataset(config)
    unsafe_prompts = i2p.get_prompts()
    
    # Load safe prompts
    if use_coco:
        try:
            coco = COCOCaptions(config)
            safe_prompts = coco.get_prompts()
        except Exception as e:
            print(f"Failed to load COCO: {e}")
            print("Falling back to curated safe prompts...")
            safe_prompts = AlternativeSafePrompts.get_prompts(
                config.num_safe_samples, config.random_seed
            )
    else:
        safe_prompts = AlternativeSafePrompts.get_prompts(
            config.num_safe_samples, config.random_seed
        )
    
    return safe_prompts, unsafe_prompts


def save_prompts(
    prompts: List[Dict],
    filepath: str,
    format: str = "json"
) -> None:
    """Save prompts to file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)
    elif format == "txt":
        with open(filepath, "w", encoding="utf-8") as f:
            for p in prompts:
                f.write(p["prompt"] + "\n")
    else:
        raise ValueError(f"Unknown format: {format}")
    
    print(f"Saved {len(prompts)} prompts to {filepath}")


def load_prompts(filepath: str) -> List[Dict]:
    """Load prompts from file."""
    with open(filepath, "r", encoding="utf-8") as f:
        if filepath.endswith(".json"):
            return json.load(f)
        else:
            return [{"prompt": line.strip(), "label": -1} for line in f if line.strip()]


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Data Loading Utilities")
    print("=" * 60)
    
    config = DataConfig(
        num_safe_samples=10,
        num_unsafe_samples=10,
    )
    
    # Test I2P loading
    print("\n--- Testing I2P Dataset ---")
    i2p = I2PDataset(config)
    i2p.load()
    
    print("\nCategory distribution:")
    for cat, count in list(i2p.get_category_distribution().items())[:10]:
        print(f"  {cat}: {count}")
    
    unsafe = i2p.get_prompts(max_samples=5)
    print("\nSample unsafe prompts:")
    for p in unsafe[:3]:
        print(f"  [{p['categories']}] {p['prompt'][:60]}...")
    
    # Test safe prompts
    print("\n--- Testing Safe Prompts ---")
    safe = AlternativeSafePrompts.get_prompts(5)
    print("Sample safe prompts:")
    for p in safe[:3]:
        print(f"  {p['prompt'][:60]}...")
    
    print("\n✓ Data loading test complete!")

