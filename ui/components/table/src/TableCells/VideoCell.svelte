<script lang="ts">
  /**
   * @copyright CEA
   * @author CEA
   * @license CECILL
   *
   * This software is a collaborative computer program whose purpose is to
   * generate and explore labeled data for computer vision applications.
   * This software is governed by the CeCILL-C license under French law and
   * abiding by the rules of distribution of free software. You can use,
   * modify and/ or redistribute the software under the terms of the CeCILL-C
   * license as circulated by CEA, CNRS and INRIA at the following URL
   *
   * http://www.cecill.info
   */

  // Exports
  export let value: string;

  // Video
  let video;
  let isHovered = false;

  function toggleHover() {
    isHovered = !isHovered;
    if (isHovered) {
      video.playbackRate = 4;
      video.play();
    } else {
      video.pause();
    }
  }
</script>

<video
  bind:this={video}
  src={value}
  loop
  muted
  class="mx-auto h-20 w-20 min-w-20 min-h-20 object-cover rounded"
  on:mouseenter={toggleHover}
  on:mouseleave={toggleHover}
/>
