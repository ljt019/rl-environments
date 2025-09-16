#!/usr/bin/env python3
"""
Generate target words/objects for twenty questions using an LLM.
"""

import asyncio
import json

from openai import AsyncOpenAI


async def generate_targets(client: AsyncOpenAI, model: str = "gpt-4o-mini", num_batches: int = 10):
    """Generate targets for twenty questions using an LLM."""

    system_prompt = """You are helping create targets for a twenty questions game. Generate diverse, concrete things that would make good targets.

Requirements:
- Mix of categories: animals, objects, places, people, food, vehicles, etc.
- Well-known things most people would recognize
- Concrete nouns (avoid abstract concepts)
- Range of difficulty (some easy like "dog", some harder like "platypus")
- No duplicates within your list
- Each should be 1-3 words max

Return ONLY a JSON array of strings, no other text."""

    user_prompt = """Generate exactly 50 diverse targets for twenty questions. Include:
- 10 animals (mix common and uncommon)
- 10 everyday objects (tools, furniture, etc.)
- 5 places/locations 
- 5 famous people (historical/contemporary)
- 5 foods/drinks
- 5 vehicles/transportation
- 10 miscellaneous interesting things

Format as JSON array: ["item1", "item2", ...]"""

    all_targets = []

    for batch in range(num_batches):
        print(f"Generating batch {batch + 1}/{num_batches}...")

        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.8,  # Higher creativity
        )

        try:
            content = response.choices[0].message.content
            targets = json.loads(content)
            all_targets.extend(targets)
            print(f"Generated {len(targets)} targets")
        except json.JSONDecodeError:
            print(f"Failed to parse batch {batch + 1}: {content[:100]}...")
            continue

    # Remove duplicates while preserving order
    seen = set()
    unique_targets = []
    for target in all_targets:
        target_lower = target.lower()
        if target_lower not in seen:
            seen.add(target_lower)
            unique_targets.append(target)

    print(f"Total unique targets: {len(unique_targets)}")
    return unique_targets


async def main():
    # Configure your LLM client
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",  # Change this
        api_key="your-api-key",  # Change this
    )

    targets = await generate_targets(client, num_batches=10)  # ~500 targets

    # Save to data.json
    with open("data.json", "w") as f:
        json.dump(targets, f, indent=2)

    print(f"Saved {len(targets)} targets to data.json")

    # Show some examples
    print("\nSample targets:")
    for target in targets[:20]:
        print(f"  - {target}")


if __name__ == "__main__":
    asyncio.run(main())
