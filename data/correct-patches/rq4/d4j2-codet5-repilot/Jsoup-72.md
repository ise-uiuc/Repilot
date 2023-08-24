# Repilot Patch

```


        if (count < 1)
            return "";
```

# Developer Patch

```
        if (count < 1)
            return "";
```

# Context

```
--- bug/Jsoup-72/src/main/java/org/jsoup/parser/CharacterReader.java

+++ fix/Jsoup-72/src/main/java/org/jsoup/parser/CharacterReader.java

@@ -425,6 +425,10 @@

         if (count > maxStringCacheLen)
             return new String(charBuf, start, count);
 
+
+        if (count < 1)
+            return "";
+
         // calculate hash:
         int hash = 0;
         int offset = start;
```

# Note

