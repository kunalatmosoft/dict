async function searchWord() {
    const wordInput = document.getElementById("word");
    const definitionDiv = document.getElementById("definition");

    if (!wordInput.value) {
        alert("Please enter a word.");
        return;
    }

    try {
        const response = await fetch(`https://api.dictionaryapi.dev/api/v2/entries/en/${wordInput.value}`);
        const data = await response.json();

        if (response.ok) {
            const wordDetails = data[0];

            if (wordDetails) {
                const meanings = wordDetails.meanings.map(meaning => {
                    const partOfSpeech = meaning.partOfSpeech || 'N/A';
                    const definitions = meaning.definitions.map(def => `- ${def.definition}`);
                    return `<strong>${partOfSpeech}</strong>: ${definitions.join('<br>')}`;
                }).join('<br><br>');

                definitionDiv.innerHTML = `<h2>${wordDetails.word}</h2>${meanings}`;
            } else {
                definitionDiv.innerText = "Definition not found.";
            }
        } else {
            definitionDiv.innerText = "Error fetching definition.";
        }
    } catch (error) {
        console.error("An error occurred:", error);
        definitionDiv.innerText = "Error occurred while processing the request.";
    }
}
