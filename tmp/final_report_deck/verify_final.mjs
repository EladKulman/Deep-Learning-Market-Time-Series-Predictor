import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const src='final report/qqq_chronos2_presentation_v2.pptx';
const p=await PresentationFile.importPptx(await FileBlob.load(src));
const out='tmp/final_report_deck/final';
await fs.mkdir(out,{recursive:true});
const comparison=[];
for(let i=0;i<p.slides.items.length;i++){
 const name=`slide-${String(i+1).padStart(2,'0')}.png`;
 const png=new Uint8Array(await (await p.slides.items[i].export({format:'png',scale:1})).arrayBuffer());
 await fs.writeFile(`${out}/${name}`,png);
 const digest=x=>crypto.createHash('sha256').update(x).digest('hex');
 comparison.push({slide:i+1,identical:digest(png)===digest(await fs.readFile(`tmp/final_report_deck/${name}`))});
}
await fs.writeFile(`${out}/render-comparison.json`,JSON.stringify(comparison,null,2));
console.log(JSON.stringify(comparison));
