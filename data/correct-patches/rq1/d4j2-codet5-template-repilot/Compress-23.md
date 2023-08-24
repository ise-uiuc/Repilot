# Repilot Patch

```
dictSize |= (coder.properties[i + 1] & 0xFFL) << (i * 8);
```

# Developer Patch

```
                dictSize |= (coder.properties[i + 1] & 0xffl) << (8 * i);
```

# Context

```
--- bug/Compress-23/src/main/java/org/apache/commons/compress/archivers/sevenz/Coders.java

+++ fix/Compress-23/src/main/java/org/apache/commons/compress/archivers/sevenz/Coders.java

@@ -109,7 +109,7 @@

             byte propsByte = coder.properties[0];
             long dictSize = coder.properties[1];
             for (int i = 1; i < 4; i++) {
-                dictSize |= (coder.properties[i + 1] << (8 * i));
+dictSize |= (coder.properties[i + 1] & 0xFFL) << (i * 8);
             }
             if (dictSize > LZMAInputStream.DICT_SIZE_MAX) {
                 throw new IOException("Dictionary larger than 4GiB maximum size");
```

# Note

