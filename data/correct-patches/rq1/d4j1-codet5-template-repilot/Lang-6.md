# Repilot Patch

```
pos += Character.charCount(Character.codePointAt(input, pt));
```

# Developer Patch

```
                pos += Character.charCount(Character.codePointAt(input, pt));
```

# Context

```
--- bug/Lang-6/src/main/java/org/apache/commons/lang3/text/translate/CharSequenceTranslator.java

+++ fix/Lang-6/src/main/java/org/apache/commons/lang3/text/translate/CharSequenceTranslator.java

@@ -92,7 +92,7 @@

 //          // contract with translators is that they have to understand codepoints 
 //          // and they just took care of a surrogate pair
             for (int pt = 0; pt < consumed; pt++) {
-                pos += Character.charCount(Character.codePointAt(input, pos));
+pos += Character.charCount(Character.codePointAt(input, pt));
             }
         }
     }
```

# Note

