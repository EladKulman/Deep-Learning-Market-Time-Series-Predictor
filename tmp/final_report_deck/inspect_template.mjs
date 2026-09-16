import fs from 'node:fs/promises';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const path='/Users/eladkulman/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.1/skills/artifact-template-simple-light-mode/assets/reference.pptx';
const p=await PresentationFile.importPptx(await FileBlob.load(path));
await fs.writeFile('tmp/final_report_deck/template_inspect.ndjson',(await p.inspect({kind:'slide,textbox,layout',maxChars:100000})).ndjson);
for(let i=0;i<p.slides.items.length;i++){
 const s=p.slides.items[i];
 await fs.writeFile(`tmp/final_report_deck/template-${i+1}.png`,new Uint8Array(await (await s.export({format:'png',scale:.7})).arrayBuffer()));
 if([0,3,4,19,20,21].includes(i))await fs.writeFile(`tmp/final_report_deck/template-${i+1}.json`,await (await s.export({format:'layout'})).text());
}
console.log('Rendered',p.slides.items.length,'template slides');
